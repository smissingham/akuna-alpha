use std::path::PathBuf;

/// Grafeo graph storage backend.
pub mod grafeo;

/// Backend-neutral graph storage mode.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphStorage {
    /// Data is kept in memory only.
    InMemory,
    /// Data is persisted at the given path.
    Persistent(PathBuf),
}

#[cfg(test)]
mod tests {
    use std::{env, process::Command, sync::Arc, time::SystemTime};

    use serde::{Deserialize, Serialize};
    use tokio::{
        task::JoinSet,
        time::{Duration, Instant, sleep},
    };

    use crate::{
        GraphError, GraphTarget,
        graph::{
            primitives::{GraphDbContext, GraphEdge, GraphNode},
            storage::{GraphStorage, grafeo::GrafeoDbContext},
        },
    };

    #[derive(Clone, Debug, GraphNode, PartialEq)]
    struct TestNode {
        #[graph(id)]
        id: String,
        #[graph(labels)]
        labels: Vec<String>,
        #[graph(name)]
        name: String,
        #[graph(description)]
        description: Option<String>,
        #[graph(metadata)]
        metadata: Option<TestMetadata>,
    }

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestMetadata {
        val: String,
    }

    #[derive(Clone, Debug, GraphEdge, PartialEq, Eq)]
    struct TestEdge {
        #[graph(source_labels)]
        source_labels: Vec<String>,
        #[graph(source)]
        source: String,
        #[graph(predicate)]
        predicate: String,
        #[graph(target)]
        target: String,
        #[graph(target_labels)]
        target_labels: Vec<String>,
    }

    const EXISTING_COUNT: usize = 20;
    const TOTAL_COUNT: usize = 30;
    const DELETE_EVERY: usize = 5;
    const UPSERT_EVERY: usize = 3;
    const HAMMER_PROCESS_COUNT: usize = 4;
    const HAMMER_NODES_PER_PROCESS: usize = 25;
    const HAMMER_READ_PASSES: usize = 25;
    const STRESS_PROCESS_COUNT: usize = 4;
    const STRESS_NODES_PER_PROCESS: usize = 250;
    const STRESS_READ_PASSES: usize = 250;
    const HAMMER_CHILD_TIMEOUT: Duration = Duration::from_secs(60);

    macro_rules! graph_db_impls {
        ($tests:ident) => {
            $tests!(
                grafeo,
                GrafeoDbContext::new_in_memory,
                GrafeoDbContext::new
            );
        };
    }

    macro_rules! graph_db_tests {
        ($name:ident, $new_in_memory:path, $new_persistent:path) => {
            mod $name {
                use super::*;

                /// Typed graph storage supports node and edge CRUD.
                #[tokio::test]
                async fn crud_lifecycle() {
                    run_crud_lifecycle($new_in_memory());

                    let db_name = db_name();
                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to initialize persisted graph db");

                    run_crud_lifecycle(graph_db);
                }

                /// Node ids are unique within labels.
                #[tokio::test]
                async fn ids_are_label_scoped() {
                    let graph_db = $new_in_memory();
                    let first = TestNode {
                        id: "same".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old".to_string(),
                        description: None,
                        metadata: Some(TestMetadata {
                            val: "old".to_string(),
                        }),
                    };
                    let second = TestNode {
                        id: first.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new".to_string(),
                        description: None,
                        metadata: Some(TestMetadata {
                            val: "new".to_string(),
                        }),
                    };

                    graph_db
                        .put_node(&first)
                        .expect("Failed to put first node");
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "Old"], &first.id)
                            .expect("Failed to count first node"),
                        1,
                    );

                    graph_db.put_node(&second).expect("Failed to upsert node");
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "Old"], &second.id)
                            .expect("Failed to count original label node"),
                        1,
                    );
                    assert_eq!(
                        graph_db
                            .node_count_by_id(&["Concept", "New"], &second.id)
                            .expect("Failed to count upserted node"),
                        1,
                    );

                    let retrieved = graph_db
                        .get_node::<TestNode>(&["Concept", "New"], &second.id)
                        .expect("Failed to read upserted node");

                    assert_eq!(retrieved, Some(second));
                }

                /// Edge endpoints are scoped by node labels.
                #[tokio::test]
                async fn edges_are_label_scoped() {
                    let graph_db = $new_in_memory();
                    let old_source = TestNode {
                        id: "source".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old source".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let new_source = TestNode {
                        id: old_source.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new source".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let old_target = TestNode {
                        id: "target".to_string(),
                        labels: vec!["Concept".to_string(), "Old".to_string()],
                        name: "old target".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let new_target = TestNode {
                        id: old_target.id.clone(),
                        labels: vec!["Concept".to_string(), "New".to_string()],
                        name: "new target".to_string(),
                        description: None,
                        metadata: None,
                    };
                    let edge = TestEdge {
                        source_labels: new_source.labels.clone(),
                        source: new_source.id.clone(),
                        predicate: "RELATES_TO".to_string(),
                        target: new_target.id.clone(),
                        target_labels: new_target.labels.clone(),
                    };
                    let old_edge = TestEdge {
                        source_labels: old_source.labels.clone(),
                        source: old_source.id.clone(),
                        predicate: edge.predicate.clone(),
                        target: old_target.id.clone(),
                        target_labels: old_target.labels.clone(),
                    };

                    graph_db
                        .put_node(&old_source)
                        .expect("Failed to put old source");
                    graph_db
                        .put_node(&new_source)
                        .expect("Failed to put new source");
                    graph_db
                        .put_node(&old_target)
                        .expect("Failed to put old target");
                    graph_db
                        .put_node(&new_target)
                        .expect("Failed to put new target");

                    graph_db
                        .put_edge(&edge)
                        .expect("Failed to put scoped edge");

                    assert!(matches!(
                        graph_db.delete_edge(&old_edge),
                        Err(GraphError::NotFound { .. })
                    ));

                    graph_db
                        .delete_edge(&edge)
                        .expect("Failed to delete scoped edge");
                }

                /// Persistent graph nodes survive context reinitialization.
                #[tokio::test]
                async fn persists_and_destroys() {
                    let db_name = db_name();
                    let item = TestNode {
                        id: "123".to_string(),
                        labels: labels_vec(),
                        name: "persisted".to_string(),
                        description: Some("persisted-val".to_string()),
                        metadata: Some(TestMetadata {
                            val: "persisted-extra".to_string(),
                        }),
                    };

                    {
                        let graph_db = $new_persistent(db_name.clone())
                            .expect("Failed to initialize persisted graph db");

                        assert!(matches!(
                            graph_db.storage(),
                            GraphStorage::Persistent(_)
                        ));

                        graph_db
                            .put_node(&item)
                            .expect("Failed to insert persisted entry");
                    }

                    let graph_db = $new_persistent(db_name.clone())
                        .expect("Failed to reinitialize persisted graph db");
                    let retrieved = graph_db
                        .get_node::<TestNode>(labels(), &item.id)
                        .expect("Failed to read persisted entry");

                    assert_eq!(retrieved, Some(item));

                    graph_db
                        .destroy()
                        .expect("Failed to clean persisted test db");

                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to reinitialize destroyed graph db");
                    let retrieved = graph_db
                        .get_node::<TestNode>(labels(), "123")
                        .expect("Failed to read from destroyed graph db");

                    assert_eq!(retrieved, None);

                    graph_db.destroy().expect(
                        "Failed to clean reinitialized destroyed graph db",
                    );
                }

                /// One shared context can handle concurrent mixed workloads safely.
                #[tokio::test]
                async fn concurrency_single_ctx() {
                    let graph_db = Arc::new($new_in_memory());
                    let mut work = JoinSet::new();

                    for index in 0..EXISTING_COUNT {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let item = TestNode {
                                id: format!("item-{index}"),
                                labels: labels_vec(),
                                name: format!("item-{index}"),
                                description: Some(format!("val-{index}")),
                                metadata: Some(TestMetadata {
                                    val: format!("extra-{index}"),
                                }),
                            };

                            graph_db.put_node(&item)
                        });
                    }

                    finish(work).await;

                    let mut work = JoinSet::new();

                    for index in 0..TOTAL_COUNT {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let id = format!("item-{index}");

                            if index < EXISTING_COUNT
                                && index % DELETE_EVERY == 0
                            {
                                return graph_db
                                    .delete_node::<TestNode>(labels(), &id)
                                    .map(|_| ());
                            }

                            let item = expected_node(index)
                                .expect("item should be present");

                            graph_db.put_node(&item).map(|_| ())
                        });
                    }

                    finish(work).await;

                    for index in 0..TOTAL_COUNT {
                        let id = format!("item-{index}");
                        let item = graph_db
                            .get_node::<TestNode>(labels(), &id)
                            .expect(
                                "Failed to read concurrently inserted entry",
                            );

                        assert_eq!(item, expected_node(index));
                    }
                }

                /// Many contexts in one process can handle concurrent mixed workloads on one storage path.
                #[tokio::test]
                async fn concurrency_multi_ctx() {
                    let db_name = db_name();
                    let mut work = JoinSet::new();

                    for index in 0..EXISTING_COUNT {
                        let db_name = db_name.clone();

                        work.spawn(async move {
                            let graph_db = $new_persistent(db_name)?;
                            let item = TestNode {
                                id: format!("item-{index}"),
                                labels: labels_vec(),
                                name: format!("item-{index}"),
                                description: Some(format!("val-{index}")),
                                metadata: Some(TestMetadata {
                                    val: format!("extra-{index}"),
                                }),
                            };

                            graph_db.put_node(&item)
                        });
                    }

                    finish(work).await;

                    let mut work = JoinSet::new();

                    for index in 0..TOTAL_COUNT {
                        let db_name = db_name.clone();

                        work.spawn(async move {
                            let graph_db = $new_persistent(db_name)?;
                            let id = format!("item-{index}");

                            if index < EXISTING_COUNT
                                && index % DELETE_EVERY == 0
                            {
                                return graph_db
                                    .delete_node::<TestNode>(labels(), &id)
                                    .map(|_| ());
                            }

                            let item = expected_node(index)
                                .expect("item should be present");

                            graph_db.put_node(&item).map(|_| ())
                        });
                    }

                    finish(work).await;

                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to open verification graph db");

                    for index in 0..TOTAL_COUNT {
                        let id = format!("item-{index}");
                        let item = graph_db
                            .get_node::<TestNode>(labels(), &id)
                            .expect(
                                "Failed to read concurrently inserted entry",
                            );

                        assert_eq!(item, expected_node(index));
                    }

                    graph_db.destroy().expect("Failed to clean test db");
                }

                /// Multiple OS processes can read and write one persistent graph.
                #[tokio::test]
                async fn concurrency_multi_process() {
                    run_concurrency_multi_process_hammer(
                        HAMMER_PROCESS_COUNT,
                        HAMMER_NODES_PER_PROCESS,
                        HAMMER_READ_PASSES,
                    )
                    .await;
                }

                /// Stress test for shared persistent graph access across OS processes.
                #[tokio::test]
                #[ignore = "manual multiprocess persistence stress test"]
                async fn concurrency_multi_process_stress() {
                    run_concurrency_multi_process_hammer(
                        STRESS_PROCESS_COUNT,
                        STRESS_NODES_PER_PROCESS,
                        STRESS_READ_PASSES,
                    )
                    .await;
                }

                async fn run_concurrency_multi_process_hammer(
                    process_count: usize,
                    nodes_per_process: usize,
                    read_passes: usize,
                ) {
                    let db_name = db_name();
                    let current_exe = env::current_exe()
                        .expect("Failed to resolve current test binary");
                    let mut children = (0..process_count)
                        .map(|process_index| {
                            Command::new(&current_exe)
                                .arg("multiprocess_hammer_child")
                                .env("AKUNA_GRAFEO_HAMMER_CHILD", "1")
                                .env("AKUNA_GRAFEO_HAMMER_DB", &db_name)
                                .env(
                                    "AKUNA_GRAFEO_HAMMER_PROCESS",
                                    process_index.to_string(),
                                )
                                .env(
                                    "AKUNA_GRAFEO_HAMMER_PROCESS_COUNT",
                                    process_count.to_string(),
                                )
                                .env(
                                    "AKUNA_GRAFEO_HAMMER_NODES",
                                    nodes_per_process.to_string(),
                                )
                                .env(
                                    "AKUNA_GRAFEO_HAMMER_READS",
                                    read_passes.to_string(),
                                )
                                .spawn()
                                .expect("Failed to spawn hammer child")
                        })
                        .collect::<Vec<_>>();

                    for child in &mut children {
                        let status = wait_for_child(child).await;

                        assert!(
                            status.success(),
                            "hammer child failed: {status}"
                        );
                    }

                    let graph_db = $new_persistent(db_name)
                        .expect("Failed to open hammer verification graph db");

                    for process_index in 0..process_count {
                        for node_index in 0..nodes_per_process {
                            let expected =
                                hammer_node(process_index, node_index);
                            let retrieved = graph_db
                                .get_node::<TestNode>(labels(), &expected.id)
                                .expect("Failed to read hammer node");

                            assert_eq!(retrieved, Some(expected));
                        }
                    }

                    graph_db.destroy().expect("Failed to clean hammer db");
                }

                /// Child entrypoint for OS process hammer test.
                #[tokio::test]
                async fn multiprocess_hammer_child() {
                    if env::var("AKUNA_GRAFEO_HAMMER_CHILD").as_deref()
                        != Ok("1")
                    {
                        return;
                    }

                    let db_name = env::var("AKUNA_GRAFEO_HAMMER_DB")
                        .expect("Missing hammer db name");
                    let process_index = env::var("AKUNA_GRAFEO_HAMMER_PROCESS")
                        .expect("Missing hammer process index")
                        .parse::<usize>()
                        .expect("Invalid hammer process index");
                    let nodes_per_process = hammer_env_usize(
                        "AKUNA_GRAFEO_HAMMER_NODES",
                        HAMMER_NODES_PER_PROCESS,
                    );
                    let read_passes = hammer_env_usize(
                        "AKUNA_GRAFEO_HAMMER_READS",
                        HAMMER_READ_PASSES,
                    );
                    let graph_db = Arc::new(
                        $new_persistent(db_name)
                            .expect("Failed to open hammer child graph db"),
                    );
                    let mut work = JoinSet::new();

                    for node_index in 0..nodes_per_process {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let item = hammer_node(process_index, node_index);

                            graph_db.put_node(&item)
                        });
                    }

                    finish(work).await;

                    let mut work = JoinSet::new();

                    for read_pass in 0..read_passes {
                        let graph_db = Arc::clone(&graph_db);

                        work.spawn(async move {
                            let node_index = read_pass % nodes_per_process;
                            let expected =
                                hammer_node(process_index, node_index);
                            let retrieved = graph_db
                                .get_node::<TestNode>(labels(), &expected.id)?;

                            assert_eq!(retrieved, Some(expected));

                            Ok(())
                        });
                    }

                    finish(work).await;

                    let graph_db = match Arc::try_unwrap(graph_db) {
                        Ok(graph_db) => graph_db,
                        Err(_) => panic!("Hammer child graph db still shared"),
                    };

                    graph_db
                        .close()
                        .expect("Failed to close hammer child graph db");
                }
            }
        };
    }

    graph_db_impls!(graph_db_tests);

    fn run_crud_lifecycle(graph_db: impl GraphDbContext) {
        let item = TestNode {
            id: "123".to_string(),
            labels: labels_vec(),
            name: "some".to_string(),
            description: Some("some-val".to_string()),
            metadata: Some(TestMetadata {
                val: "some-extra".to_string(),
            }),
        };

        graph_db.put_node(&item).expect("Failed to put entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &item.id)
            .expect("Failed to read entry");

        assert_eq!(retrieved, Some(item.clone()));

        let updated = TestNode {
            id: item.id.clone(),
            labels: labels_vec(),
            name: "updated".to_string(),
            description: Some("updated-val".to_string()),
            metadata: Some(TestMetadata {
                val: "updated-extra".to_string(),
            }),
        };

        graph_db.put_node(&updated).expect("Failed to update entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to read updated entry");

        assert_eq!(retrieved, Some(updated.clone()));

        let inserted = TestNode {
            id: "456".to_string(),
            labels: labels_vec(),
            name: "inserted".to_string(),
            description: Some("inserted-val".to_string()),
            metadata: Some(TestMetadata {
                val: "inserted-extra".to_string(),
            }),
        };
        graph_db
            .put_node(&inserted)
            .expect("Failed to put missing entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &inserted.id)
            .expect("Failed to read inserted entry");

        assert_eq!(retrieved, Some(inserted.clone()));

        let edge = TestEdge {
            source_labels: updated.labels.clone(),
            source: updated.id.clone(),
            predicate: "RELATES_TO".to_string(),
            target: inserted.id.clone(),
            target_labels: inserted.labels.clone(),
        };

        graph_db.put_edge(&edge).expect("Failed to put edge");
        graph_db.delete_edge(&edge).expect("Failed to delete edge");

        let missing_edge_delete = graph_db.delete_edge(&edge);

        assert!(matches!(
            missing_edge_delete,
            Err(GraphError::NotFound {
                target: GraphTarget::Edge { predicate, .. }
            }) if predicate == edge.predicate()
        ));

        graph_db
            .delete_node::<TestNode>(labels(), &inserted.id)
            .expect("Failed to delete inserted entry");

        graph_db
            .delete_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to delete entry");

        let retrieved = graph_db
            .get_node::<TestNode>(labels(), &updated.id)
            .expect("Failed to read deleted entry");

        assert_eq!(retrieved, None);

        let missing_delete =
            graph_db.delete_node::<TestNode>(labels(), &updated.id);

        assert!(matches!(
            missing_delete,
            Err(GraphError::NotFound {
                target: GraphTarget::Node { id, .. }
            }) if id == updated.id
        ));

        graph_db.destroy().expect("Failed to clean test db");
    }

    async fn finish(mut work: JoinSet<Result<(), GraphError>>) {
        while let Some(result) = work.join_next().await {
            result
                .expect("Concurrent task panicked")
                .expect("Concurrent task failed");
        }
    }

    async fn wait_for_child(
        child: &mut std::process::Child,
    ) -> std::process::ExitStatus {
        let deadline = Instant::now() + HAMMER_CHILD_TIMEOUT;

        loop {
            if let Some(status) =
                child.try_wait().expect("Failed to wait for hammer child")
            {
                return status;
            }

            if Instant::now() >= deadline {
                let _ = child.kill();
                panic!("Timed out waiting for hammer child");
            }

            sleep(Duration::from_millis(25)).await;
        }
    }

    fn hammer_env_usize(name: &str, default: usize) -> usize {
        env::var(name)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(default)
    }

    fn expected_node(index: usize) -> Option<TestNode> {
        if index < EXISTING_COUNT && index.is_multiple_of(DELETE_EVERY) {
            return None;
        }

        let description = if index >= EXISTING_COUNT {
            format!("inserted-{index}")
        } else if index.is_multiple_of(UPSERT_EVERY) {
            format!("upserted-{index}")
        } else {
            format!("updated-{index}")
        };

        Some(TestNode {
            id: format!("item-{index}"),
            labels: labels_vec(),
            name: format!("item-{index}"),
            description: Some(description),
            metadata: Some(TestMetadata {
                val: format!("extra-{index}"),
            }),
        })
    }

    fn labels() -> &'static [&'static str] {
        &["Concept", "Item"]
    }

    fn labels_vec() -> Vec<String> {
        labels().iter().map(|label| (*label).to_string()).collect()
    }

    fn hammer_node(process_index: usize, node_index: usize) -> TestNode {
        TestNode {
            id: hammer_node_id(process_index, node_index),
            labels: labels_vec(),
            name: format!("hammer-{process_index}-{node_index}"),
            description: Some(format!(
                "hammer-val-{process_index}-{node_index}"
            )),
            metadata: Some(TestMetadata {
                val: format!("hammer-extra-{process_index}-{node_index}"),
            }),
        }
    }

    fn hammer_node_id(process_index: usize, node_index: usize) -> String {
        format!("hammer-{process_index}-{node_index}")
    }

    fn db_name() -> String {
        format!(
            "graphdb-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("System clock should be after Unix epoch")
                .as_nanos(),
        )
    }
}
