use std::ops::Range;

/// Tree-sitter syntax node range used for structured code extraction.
pub(crate) struct CodePartRange {
    /// Node kind reported by tree-sitter.
    pub(crate) kind: String,
    /// Byte range in source content.
    pub(crate) range: Range<usize>,
}

/// Extract top-level named syntax node ranges for code/document structure.
pub(crate) fn code_part_ranges(
    content: &str,
    file_extension: Option<&str>,
) -> Option<Vec<CodePartRange>> {
    let language = language_for_extension(file_extension?)?;
    let tree = syntax_tree(content, language)?;

    if tree.root_node().has_error() {
        return None;
    }

    let mut ranges = Vec::new();
    collect_code_part_ranges(tree.root_node(), &mut ranges);

    (!ranges.is_empty()).then_some(ranges)
}

/// Collect useful named syntax nodes with hierarchy metadata.
fn collect_code_part_ranges(
    node: tree_sitter::Node<'_>,
    ranges: &mut Vec<CodePartRange>,
) -> bool {
    if node.parent().is_some()
        && is_code_part_node(node)
        && (is_function_depth_node(node) || !has_child_code_parts(node))
    {
        ranges.push(CodePartRange {
            kind: node.kind().to_owned(),
            range: node.byte_range(),
        });
        return true;
    }

    let start_len = ranges.len();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        collect_code_part_ranges(child, ranges);
    }

    ranges.len() > start_len
}

/// Returns whether a node contains child code parts.
fn has_child_code_parts(node: tree_sitter::Node<'_>) -> bool {
    let mut cursor = node.walk();
    node.named_children(&mut cursor).any(contains_code_part)
}

/// Returns whether a node or descendant is a code part.
fn contains_code_part(node: tree_sitter::Node<'_>) -> bool {
    if is_code_part_node(node) {
        return true;
    }

    let mut cursor = node.walk();
    node.named_children(&mut cursor).any(contains_code_part)
}

/// Returns whether a named node is useful as an extraction part.
fn is_code_part_node(node: tree_sitter::Node<'_>) -> bool {
    if !node.is_named() || node.byte_range().is_empty() {
        return false;
    }

    let kind = node.kind();
    if is_code_container_node(kind) {
        return false;
    }

    is_function_depth_node(node)
        || kind.ends_with("declaration")
        || kind.ends_with("definition")
        || kind.ends_with("item")
        || kind.ends_with("class")
        || matches!(
            kind,
            "constructor_declaration"
                | "field_declaration"
                | "impl_item"
                | "interface_declaration"
                | "struct_item"
        )
}

/// Returns whether node is deepest semantic code part worth descending to.
fn is_function_depth_node(node: tree_sitter::Node<'_>) -> bool {
    let kind = node.kind();
    kind.contains("function")
        || kind.contains("method")
        || kind.contains("constructor")
        || matches!(kind, "lambda_expression" | "record_declaration")
}

/// Returns whether node exists only to group child syntax.
fn is_code_container_node(kind: &str) -> bool {
    matches!(
        kind,
        "class_body" | "declaration_list" | "module" | "program"
    )
}

/// Grammar-aware chunking via tree-sitter.
///
/// Returns `None` when no language matches, parsing fails, or the
/// syntax tree contains errors.
pub(super) fn chunk_content<'a>(
    content: &'a str,
    delimiters: Option<&[u8]>,
    target_size: usize,
    file_extension: Option<&str>,
) -> Option<Vec<&'a str>> {
    let language = language_for_extension(file_extension?)?;
    let tree = syntax_tree(content, language)?;

    if tree.root_node().has_error() {
        return None;
    }

    let mut ranges = Vec::new();
    collect_chunk_ranges(tree.root_node(), target_size, &mut ranges);

    if ranges.is_empty() {
        return None;
    }

    let mut chunks = Vec::new();
    let mut chunk_start = 0;
    let mut chunk_end = ranges[0].end;

    for range in ranges.into_iter().skip(1) {
        if range.end - chunk_start > target_size {
            push_chunk(
                &mut chunks,
                content,
                chunk_start,
                chunk_end,
                delimiters,
                target_size,
            );
            chunk_start = chunk_end;
        }

        chunk_end = range.end;
    }

    push_chunk(
        &mut chunks,
        content,
        chunk_start,
        content.len(),
        delimiters,
        target_size,
    );

    Some(chunks)
}

/// Walks the syntax tree collecting byte ranges that fit target size.
fn collect_chunk_ranges(
    node: tree_sitter::Node<'_>,
    target_size: usize,
    ranges: &mut Vec<Range<usize>>,
) {
    let mut cursor = node.walk();
    let children = node.named_children(&mut cursor).collect::<Vec<_>>();

    if children.is_empty() {
        ranges.push(node.byte_range());
        return;
    }

    if node.parent().is_some() && node.byte_range().len() <= target_size {
        ranges.push(node.byte_range());
        return;
    }

    for child in children {
        collect_chunk_ranges(child, target_size, ranges);
    }
}

/// Parses content into a tree-sitter syntax tree.
fn syntax_tree(
    content: &str,
    language: tree_sitter::Language,
) -> Option<tree_sitter::Tree> {
    let mut parser = tree_sitter::Parser::new();

    if parser.set_language(&language).is_err() {
        return None;
    }

    parser.parse(content, None)
}

/// Pushes a content slice, sub-chunking when it exceeds target size.
fn push_chunk<'a>(
    chunks: &mut Vec<&'a str>,
    content: &'a str,
    start: usize,
    end: usize,
    delimiters: Option<&[u8]>,
    target_size: usize,
) {
    let text = &content[start..end];

    if text.len() <= target_size {
        chunks.push(text);
        return;
    }

    chunks.extend(super::chunk_with_delimiters(text, delimiters, target_size));
}

/// Maps a file extension to its tree-sitter language, if supported.
fn language_for_extension(
    file_extension: &str,
) -> Option<tree_sitter::Language> {
    match file_extension {
        "bash" | "sh" => Some(tree_sitter_bash::LANGUAGE.into()),
        "c" => Some(tree_sitter_c::LANGUAGE.into()),
        "cpp" => Some(tree_sitter_cpp::LANGUAGE.into()),
        "containerfile" | "dockerfile" => {
            Some(tree_sitter_containerfile::LANGUAGE.into())
        }
        "cs" => Some(tree_sitter_c_sharp::LANGUAGE.into()),
        "css" => Some(tree_sitter_css::LANGUAGE.into()),
        "dart" => Some(tree_sitter_dart::LANGUAGE.into()),
        "go" => Some(tree_sitter_go::LANGUAGE.into()),
        "graphql" | "gql" => Some(tree_sitter_graphql::LANGUAGE.into()),
        "groovy" | "gradle" => Some(tree_sitter_groovy::LANGUAGE.into()),
        "hcl" | "tf" | "terraform" => Some(tree_sitter_hcl::LANGUAGE.into()),
        "html" => Some(tree_sitter_html::LANGUAGE.into()),
        "java" => Some(tree_sitter_java::LANGUAGE.into()),
        "js" | "jsx" => Some(tree_sitter_javascript::LANGUAGE.into()),
        "json" => Some(tree_sitter_json::LANGUAGE.into()),
        "lua" => Some(tree_sitter_lua::LANGUAGE.into()),
        "md" | "markdown" => Some(tree_sitter_md::LANGUAGE.into()),
        "php" => Some(tree_sitter_php::LANGUAGE_PHP.into()),
        "py" => Some(tree_sitter_python::LANGUAGE.into()),
        "rb" | "ruby" => Some(tree_sitter_ruby::LANGUAGE.into()),
        "rs" => Some(tree_sitter_rust::LANGUAGE.into()),
        "scala" | "sc" => Some(tree_sitter_scala::LANGUAGE.into()),
        "scss" => Some(tree_sitter_scss::language()),
        "sql" => Some(tree_sitter_sequel::LANGUAGE.into()),
        "svelte" => Some(tree_sitter_svelte_next::LANGUAGE.into()),
        "swift" => Some(tree_sitter_swift::LANGUAGE.into()),
        "ts" => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
        "tsx" => Some(tree_sitter_typescript::LANGUAGE_TSX.into()),
        "vue" => Some(tree_sitter_vue_updated::language()),
        "yaml" | "yml" => Some(tree_sitter_yaml::LANGUAGE.into()),
        _ => None,
    }
}
