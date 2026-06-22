// Generated from ONNX "/tmp/opencode-build/akuna-ocr-paddle-codegen/patched/medium_rec.onnx" by burn-onnx
use burn::nn::BatchNorm;
use burn::nn::BatchNormConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::pool::AvgPool2d;
use burn::nn::pool::AvgPool2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::prelude::*;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;

#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    constant2: burn::module::Param<Tensor<B, 4>>,
    conv2d2: Conv2d<B>,
    constant4: burn::module::Param<Tensor<B, 4>>,
    conv2d3: Conv2d<B>,
    constant6: burn::module::Param<Tensor<B, 4>>,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d<B>,
    constant8: burn::module::Param<Tensor<B, 4>>,
    conv2d5: Conv2d<B>,
    constant10: burn::module::Param<Tensor<B, 4>>,
    conv2d6: Conv2d<B>,
    constant12: burn::module::Param<Tensor<B, 4>>,
    conv2d7: Conv2d<B>,
    constant14: burn::module::Param<Tensor<B, 4>>,
    conv2d8: Conv2d<B>,
    constant16: burn::module::Param<Tensor<B, 4>>,
    conv2d9: Conv2d<B>,
    constant18: burn::module::Param<Tensor<B, 4>>,
    constant19: burn::module::Param<Tensor<B, 1>>,
    constant20: burn::module::Param<Tensor<B, 1>>,
    constant21: burn::module::Param<Tensor<B, 1>>,
    conv2d10: Conv2d<B>,
    constant23: burn::module::Param<Tensor<B, 4>>,
    conv2d11: Conv2d<B>,
    constant25: burn::module::Param<Tensor<B, 4>>,
    conv2d12: Conv2d<B>,
    constant27: burn::module::Param<Tensor<B, 4>>,
    constant28: burn::module::Param<Tensor<B, 1>>,
    constant29: burn::module::Param<Tensor<B, 1>>,
    constant30: burn::module::Param<Tensor<B, 1>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule1<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d1 = Conv2dConfig::new([3, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant2: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 64, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 64, 1, 1].into(),
            );
        let conv2d2 = Conv2dConfig::new([64, 32], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant4: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 32, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 32, 1, 1].into(),
            );
        let conv2d3 = Conv2dConfig::new([32, 64], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant6: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 64, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 64, 1, 1].into(),
            );
        let maxpool2d1 = MaxPool2dConfig::new([2, 2])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_ceil_mode(false)
            .init();
        let conv2d4 = Conv2dConfig::new([128, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant8: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 64, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 64, 1, 1].into(),
            );
        let conv2d5 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant10: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d6 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let constant12: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d7 = Conv2dConfig::new([128, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant14: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 32, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 32, 1, 1].into(),
            );
        let conv2d8 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant16: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d9 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant18: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let constant19: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant20: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant21: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d10 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant23: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d11 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(false)
            .init(device);
        let constant25: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d12 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant27: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let constant28: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant29: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant30: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        Self {
            conv2d1,
            constant2,
            conv2d2,
            constant4,
            conv2d3,
            constant6,
            maxpool2d1,
            conv2d4,
            constant8,
            conv2d5,
            constant10,
            conv2d6,
            constant12,
            conv2d7,
            constant14,
            conv2d8,
            constant16,
            conv2d9,
            constant18,
            constant19,
            constant20,
            constant21,
            conv2d10,
            constant23,
            conv2d11,
            constant25,
            conv2d12,
            constant27,
            constant28,
            constant29,
            constant30,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d1_out1 = self.conv2d1.forward(x);
        let constant2_out1 = self.constant2.val();
        let add1_out1 = conv2d1_out1.add(constant2_out1);
        let relu1_out1 = burn::tensor::activation::relu(add1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let constant4_out1 = self.constant4.val();
        let add2_out1 = conv2d2_out1.add(constant4_out1);
        let relu2_out1 = burn::tensor::activation::relu(add2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let constant6_out1 = self.constant6.val();
        let add3_out1 = conv2d3_out1.add(constant6_out1);
        let relu3_out1 = burn::tensor::activation::relu(add3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);
        let concat1_out1 =
            burn::tensor::Tensor::cat([maxpool2d1_out1, relu3_out1].into(), 1);
        let conv2d4_out1 = self.conv2d4.forward(concat1_out1);
        let constant8_out1 = self.constant8.val();
        let add4_out1 = conv2d4_out1.add(constant8_out1);
        let relu4_out1 = burn::tensor::activation::relu(add4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let constant10_out1 = self.constant10.val();
        let add5_out1 = conv2d5_out1.add(constant10_out1);
        let relu5_out1 = burn::tensor::activation::relu(add5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let constant12_out1 = self.constant12.val();
        let add6_out1 = conv2d6_out1.add(constant12_out1);
        let reducemean1_out1 =
            { add6_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d7_out1 = self.conv2d7.forward(reducemean1_out1);
        let constant14_out1 = self.constant14.val();
        let add7_out1 = conv2d7_out1.add(constant14_out1);
        let relu6_out1 = burn::tensor::activation::relu(add7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let constant16_out1 = self.constant16.val();
        let add8_out1 = conv2d8_out1.add(constant16_out1);
        let hardsigmoid1_out1 = burn::tensor::activation::hard_sigmoid(
            add8_out1,
            0.16666670143604279,
            0.5,
        );
        let mul1_out1 = add6_out1.mul(hardsigmoid1_out1);
        let conv2d9_out1 = self.conv2d9.forward(mul1_out1.clone());
        let constant18_out1 = self.constant18.val();
        let add9_out1 = conv2d9_out1.add(constant18_out1);
        let constant19_out1 = self.constant19.val();
        let div1_out1 = add9_out1
            .clone()
            .div((constant19_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf1_out1 = div1_out1.erf();
        let constant20_out1 = self.constant20.val();
        let add10_out1 = erf1_out1
            .add((constant20_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul2_out1 = add9_out1.mul(add10_out1);
        let constant21_out1 = self.constant21.val();
        let mul3_out1 = mul2_out1
            .mul((constant21_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d10_out1 = self.conv2d10.forward(mul3_out1);
        let constant23_out1 = self.constant23.val();
        let add11_out1 = conv2d10_out1.add(constant23_out1);
        let add12_out1 = mul1_out1.add(add11_out1);
        let conv2d11_out1 = self.conv2d11.forward(add12_out1);
        let constant25_out1 = self.constant25.val();
        let add13_out1 = conv2d11_out1.add(constant25_out1);
        let conv2d12_out1 = self.conv2d12.forward(add13_out1);
        let constant27_out1 = self.constant27.val();
        let add14_out1 = conv2d12_out1.add(constant27_out1);
        let constant28_out1 = self.constant28.val();
        let div2_out1 = add14_out1
            .clone()
            .div((constant28_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf2_out1 = div2_out1.erf();
        let constant29_out1 = self.constant29.val();
        let add15_out1 = erf2_out1
            .add((constant29_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul4_out1 = add14_out1.mul(add15_out1);
        let constant30_out1 = self.constant30.val();
        let mul5_out1 = mul4_out1
            .mul((constant30_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        mul5_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    conv2d13: Conv2d<B>,
    constant32: burn::module::Param<Tensor<B, 4>>,
    conv2d14: Conv2d<B>,
    constant34: burn::module::Param<Tensor<B, 4>>,
    conv2d15: Conv2d<B>,
    constant36: burn::module::Param<Tensor<B, 4>>,
    constant37: burn::module::Param<Tensor<B, 1>>,
    constant38: burn::module::Param<Tensor<B, 1>>,
    constant39: burn::module::Param<Tensor<B, 1>>,
    conv2d16: Conv2d<B>,
    constant41: burn::module::Param<Tensor<B, 4>>,
    conv2d17: Conv2d<B>,
    constant43: burn::module::Param<Tensor<B, 4>>,
    conv2d18: Conv2d<B>,
    constant45: burn::module::Param<Tensor<B, 4>>,
    conv2d19: Conv2d<B>,
    constant47: burn::module::Param<Tensor<B, 4>>,
    conv2d20: Conv2d<B>,
    constant49: burn::module::Param<Tensor<B, 4>>,
    constant50: burn::module::Param<Tensor<B, 1>>,
    constant51: burn::module::Param<Tensor<B, 1>>,
    constant52: burn::module::Param<Tensor<B, 1>>,
    conv2d21: Conv2d<B>,
    constant54: burn::module::Param<Tensor<B, 4>>,
    conv2d22: Conv2d<B>,
    constant56: burn::module::Param<Tensor<B, 4>>,
    conv2d23: Conv2d<B>,
    constant58: burn::module::Param<Tensor<B, 4>>,
    constant59: burn::module::Param<Tensor<B, 1>>,
    constant60: burn::module::Param<Tensor<B, 1>>,
    constant61: burn::module::Param<Tensor<B, 1>>,
    conv2d24: Conv2d<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d13 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant32: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d14 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let constant34: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d15 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant36: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let constant37: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant38: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant39: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d16 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant41: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d17 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let constant43: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d18 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant45: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 64, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 64, 1, 1].into(),
            );
        let conv2d19 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant47: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d20 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant49: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let constant50: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant51: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant52: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d21 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant54: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d22 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(false)
            .init(device);
        let constant56: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 256, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 256, 1, 1].into(),
            );
        let conv2d23 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant58: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let constant59: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant60: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant61: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d24 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        Self {
            conv2d13,
            constant32,
            conv2d14,
            constant34,
            conv2d15,
            constant36,
            constant37,
            constant38,
            constant39,
            conv2d16,
            constant41,
            conv2d17,
            constant43,
            conv2d18,
            constant45,
            conv2d19,
            constant47,
            conv2d20,
            constant49,
            constant50,
            constant51,
            constant52,
            conv2d21,
            constant54,
            conv2d22,
            constant56,
            conv2d23,
            constant58,
            constant59,
            constant60,
            constant61,
            conv2d24,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, mul5_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d13_out1 = self.conv2d13.forward(mul5_out1);
        let constant32_out1 = self.constant32.val();
        let add16_out1 = conv2d13_out1.add(constant32_out1);
        let conv2d14_out1 = self.conv2d14.forward(add16_out1);
        let constant34_out1 = self.constant34.val();
        let add17_out1 = conv2d14_out1.add(constant34_out1);
        let conv2d15_out1 = self.conv2d15.forward(add17_out1.clone());
        let constant36_out1 = self.constant36.val();
        let add18_out1 = conv2d15_out1.add(constant36_out1);
        let constant37_out1 = self.constant37.val();
        let div3_out1 = add18_out1
            .clone()
            .div((constant37_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf3_out1 = div3_out1.erf();
        let constant38_out1 = self.constant38.val();
        let add19_out1 = erf3_out1
            .add((constant38_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul6_out1 = add18_out1.mul(add19_out1);
        let constant39_out1 = self.constant39.val();
        let mul7_out1 = mul6_out1
            .mul((constant39_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d16_out1 = self.conv2d16.forward(mul7_out1);
        let constant41_out1 = self.constant41.val();
        let add20_out1 = conv2d16_out1.add(constant41_out1);
        let add21_out1 = add17_out1.add(add20_out1);
        let conv2d17_out1 = self.conv2d17.forward(add21_out1);
        let constant43_out1 = self.constant43.val();
        let add22_out1 = conv2d17_out1.add(constant43_out1);
        let reducemean2_out1 =
            { add22_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d18_out1 = self.conv2d18.forward(reducemean2_out1);
        let constant45_out1 = self.constant45.val();
        let add23_out1 = conv2d18_out1.add(constant45_out1);
        let relu7_out1 = burn::tensor::activation::relu(add23_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu7_out1);
        let constant47_out1 = self.constant47.val();
        let add24_out1 = conv2d19_out1.add(constant47_out1);
        let hardsigmoid2_out1 = burn::tensor::activation::hard_sigmoid(
            add24_out1,
            0.16666670143604279,
            0.5,
        );
        let mul8_out1 = add22_out1.mul(hardsigmoid2_out1);
        let conv2d20_out1 = self.conv2d20.forward(mul8_out1.clone());
        let constant49_out1 = self.constant49.val();
        let add25_out1 = conv2d20_out1.add(constant49_out1);
        let constant50_out1 = self.constant50.val();
        let div4_out1 = add25_out1
            .clone()
            .div((constant50_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf4_out1 = div4_out1.erf();
        let constant51_out1 = self.constant51.val();
        let add26_out1 = erf4_out1
            .add((constant51_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul9_out1 = add25_out1.mul(add26_out1);
        let constant52_out1 = self.constant52.val();
        let mul10_out1 = mul9_out1
            .mul((constant52_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d21_out1 = self.conv2d21.forward(mul10_out1);
        let constant54_out1 = self.constant54.val();
        let add27_out1 = conv2d21_out1.add(constant54_out1);
        let add28_out1 = mul8_out1.add(add27_out1);
        let conv2d22_out1 = self.conv2d22.forward(add28_out1);
        let constant56_out1 = self.constant56.val();
        let add29_out1 = conv2d22_out1.add(constant56_out1);
        let conv2d23_out1 = self.conv2d23.forward(add29_out1);
        let constant58_out1 = self.constant58.val();
        let add30_out1 = conv2d23_out1.add(constant58_out1);
        let constant59_out1 = self.constant59.val();
        let div5_out1 = add30_out1
            .clone()
            .div((constant59_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf5_out1 = div5_out1.erf();
        let constant60_out1 = self.constant60.val();
        let add31_out1 = erf5_out1
            .add((constant60_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul11_out1 = add30_out1.mul(add31_out1);
        let constant61_out1 = self.constant61.val();
        let mul12_out1 = mul11_out1
            .mul((constant61_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d24_out1 = self.conv2d24.forward(mul12_out1);
        conv2d24_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule3<B: Backend> {
    constant63: burn::module::Param<Tensor<B, 4>>,
    conv2d25: Conv2d<B>,
    constant65: burn::module::Param<Tensor<B, 4>>,
    conv2d26: Conv2d<B>,
    constant67: burn::module::Param<Tensor<B, 4>>,
    conv2d27: Conv2d<B>,
    constant69: burn::module::Param<Tensor<B, 4>>,
    conv2d28: Conv2d<B>,
    constant71: burn::module::Param<Tensor<B, 4>>,
    constant72: burn::module::Param<Tensor<B, 1>>,
    constant73: burn::module::Param<Tensor<B, 1>>,
    constant74: burn::module::Param<Tensor<B, 1>>,
    conv2d29: Conv2d<B>,
    constant76: burn::module::Param<Tensor<B, 4>>,
    conv2d30: Conv2d<B>,
    constant78: burn::module::Param<Tensor<B, 4>>,
    conv2d31: Conv2d<B>,
    constant80: burn::module::Param<Tensor<B, 4>>,
    constant81: burn::module::Param<Tensor<B, 1>>,
    constant82: burn::module::Param<Tensor<B, 1>>,
    constant83: burn::module::Param<Tensor<B, 1>>,
    conv2d32: Conv2d<B>,
    constant85: burn::module::Param<Tensor<B, 4>>,
    conv2d33: Conv2d<B>,
    constant87: burn::module::Param<Tensor<B, 4>>,
    conv2d34: Conv2d<B>,
    constant89: burn::module::Param<Tensor<B, 4>>,
    conv2d35: Conv2d<B>,
    constant91: burn::module::Param<Tensor<B, 4>>,
    conv2d36: Conv2d<B>,
    constant93: burn::module::Param<Tensor<B, 4>>,
    constant94: burn::module::Param<Tensor<B, 1>>,
    constant95: burn::module::Param<Tensor<B, 1>>,
    constant96: burn::module::Param<Tensor<B, 1>>,
    conv2d37: Conv2d<B>,
    constant98: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule3<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let constant63: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d25 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant65: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d26 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant67: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d27 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant69: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d28 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant71: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant72: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant73: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant74: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d29 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant76: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d30 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant78: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d31 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant80: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant81: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant82: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant83: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d32 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant85: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d33 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant87: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d34 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant89: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d35 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant91: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d36 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant93: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant94: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant95: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant96: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d37 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant98: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        Self {
            constant63,
            conv2d25,
            constant65,
            conv2d26,
            constant67,
            conv2d27,
            constant69,
            conv2d28,
            constant71,
            constant72,
            constant73,
            constant74,
            conv2d29,
            constant76,
            conv2d30,
            constant78,
            conv2d31,
            constant80,
            constant81,
            constant82,
            constant83,
            conv2d32,
            constant85,
            conv2d33,
            constant87,
            conv2d34,
            constant89,
            conv2d35,
            constant91,
            conv2d36,
            constant93,
            constant94,
            constant95,
            constant96,
            conv2d37,
            constant98,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, conv2d24_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let constant63_out1 = self.constant63.val();
        let add32_out1 = conv2d24_out1.add(constant63_out1);
        let conv2d25_out1 = self.conv2d25.forward(add32_out1);
        let constant65_out1 = self.constant65.val();
        let add33_out1 = conv2d25_out1.add(constant65_out1);
        let reducemean3_out1 =
            { add33_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d26_out1 = self.conv2d26.forward(reducemean3_out1);
        let constant67_out1 = self.constant67.val();
        let add34_out1 = conv2d26_out1.add(constant67_out1);
        let relu8_out1 = burn::tensor::activation::relu(add34_out1);
        let conv2d27_out1 = self.conv2d27.forward(relu8_out1);
        let constant69_out1 = self.constant69.val();
        let add35_out1 = conv2d27_out1.add(constant69_out1);
        let hardsigmoid3_out1 = burn::tensor::activation::hard_sigmoid(
            add35_out1,
            0.16666670143604279,
            0.5,
        );
        let mul13_out1 = add33_out1.mul(hardsigmoid3_out1);
        let conv2d28_out1 = self.conv2d28.forward(mul13_out1.clone());
        let constant71_out1 = self.constant71.val();
        let add36_out1 = conv2d28_out1.add(constant71_out1);
        let constant72_out1 = self.constant72.val();
        let div6_out1 = add36_out1
            .clone()
            .div((constant72_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf6_out1 = div6_out1.erf();
        let constant73_out1 = self.constant73.val();
        let add37_out1 = erf6_out1
            .add((constant73_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul14_out1 = add36_out1.mul(add37_out1);
        let constant74_out1 = self.constant74.val();
        let mul15_out1 = mul14_out1
            .mul((constant74_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d29_out1 = self.conv2d29.forward(mul15_out1);
        let constant76_out1 = self.constant76.val();
        let add38_out1 = conv2d29_out1.add(constant76_out1);
        let add39_out1 = mul13_out1.add(add38_out1);
        let conv2d30_out1 = self.conv2d30.forward(add39_out1);
        let constant78_out1 = self.constant78.val();
        let add40_out1 = conv2d30_out1.add(constant78_out1);
        let conv2d31_out1 = self.conv2d31.forward(add40_out1.clone());
        let constant80_out1 = self.constant80.val();
        let add41_out1 = conv2d31_out1.add(constant80_out1);
        let constant81_out1 = self.constant81.val();
        let div7_out1 = add41_out1
            .clone()
            .div((constant81_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf7_out1 = div7_out1.erf();
        let constant82_out1 = self.constant82.val();
        let add42_out1 = erf7_out1
            .add((constant82_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul16_out1 = add41_out1.mul(add42_out1);
        let constant83_out1 = self.constant83.val();
        let mul17_out1 = mul16_out1
            .mul((constant83_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d32_out1 = self.conv2d32.forward(mul17_out1);
        let constant85_out1 = self.constant85.val();
        let add43_out1 = conv2d32_out1.add(constant85_out1);
        let add44_out1 = add40_out1.add(add43_out1);
        let conv2d33_out1 = self.conv2d33.forward(add44_out1);
        let constant87_out1 = self.constant87.val();
        let add45_out1 = conv2d33_out1.add(constant87_out1);
        let reducemean4_out1 =
            { add45_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d34_out1 = self.conv2d34.forward(reducemean4_out1);
        let constant89_out1 = self.constant89.val();
        let add46_out1 = conv2d34_out1.add(constant89_out1);
        let relu9_out1 = burn::tensor::activation::relu(add46_out1);
        let conv2d35_out1 = self.conv2d35.forward(relu9_out1);
        let constant91_out1 = self.constant91.val();
        let add47_out1 = conv2d35_out1.add(constant91_out1);
        let hardsigmoid4_out1 = burn::tensor::activation::hard_sigmoid(
            add47_out1,
            0.16666670143604279,
            0.5,
        );
        let mul18_out1 = add45_out1.mul(hardsigmoid4_out1);
        let conv2d36_out1 = self.conv2d36.forward(mul18_out1.clone());
        let constant93_out1 = self.constant93.val();
        let add48_out1 = conv2d36_out1.add(constant93_out1);
        let constant94_out1 = self.constant94.val();
        let div8_out1 = add48_out1
            .clone()
            .div((constant94_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf8_out1 = div8_out1.erf();
        let constant95_out1 = self.constant95.val();
        let add49_out1 = erf8_out1
            .add((constant95_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul19_out1 = add48_out1.mul(add49_out1);
        let constant96_out1 = self.constant96.val();
        let mul20_out1 = mul19_out1
            .mul((constant96_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d37_out1 = self.conv2d37.forward(mul20_out1);
        let constant98_out1 = self.constant98.val();
        let add50_out1 = conv2d37_out1.add(constant98_out1);
        let add51_out1 = mul18_out1.add(add50_out1);
        add51_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule4<B: Backend> {
    conv2d38: Conv2d<B>,
    constant100: burn::module::Param<Tensor<B, 4>>,
    conv2d39: Conv2d<B>,
    constant102: burn::module::Param<Tensor<B, 4>>,
    constant103: burn::module::Param<Tensor<B, 1>>,
    constant104: burn::module::Param<Tensor<B, 1>>,
    constant105: burn::module::Param<Tensor<B, 1>>,
    conv2d40: Conv2d<B>,
    constant107: burn::module::Param<Tensor<B, 4>>,
    conv2d41: Conv2d<B>,
    constant109: burn::module::Param<Tensor<B, 4>>,
    conv2d42: Conv2d<B>,
    constant111: burn::module::Param<Tensor<B, 4>>,
    conv2d43: Conv2d<B>,
    constant113: burn::module::Param<Tensor<B, 4>>,
    conv2d44: Conv2d<B>,
    constant115: burn::module::Param<Tensor<B, 4>>,
    constant116: burn::module::Param<Tensor<B, 1>>,
    constant117: burn::module::Param<Tensor<B, 1>>,
    constant118: burn::module::Param<Tensor<B, 1>>,
    conv2d45: Conv2d<B>,
    constant120: burn::module::Param<Tensor<B, 4>>,
    conv2d46: Conv2d<B>,
    constant122: burn::module::Param<Tensor<B, 4>>,
    conv2d47: Conv2d<B>,
    constant124: burn::module::Param<Tensor<B, 4>>,
    constant125: burn::module::Param<Tensor<B, 1>>,
    constant126: burn::module::Param<Tensor<B, 1>>,
    constant127: burn::module::Param<Tensor<B, 1>>,
    conv2d48: Conv2d<B>,
    constant129: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule4<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d38 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant100: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d39 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant102: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant103: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant104: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant105: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d40 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant107: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d41 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant109: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d42 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant111: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 128, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 128, 1, 1].into(),
            );
        let conv2d43 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant113: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d44 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant115: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant116: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant117: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant118: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d45 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant120: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d46 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant122: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d47 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant124: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant125: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant126: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant127: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d48 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant129: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        Self {
            conv2d38,
            constant100,
            conv2d39,
            constant102,
            constant103,
            constant104,
            constant105,
            conv2d40,
            constant107,
            conv2d41,
            constant109,
            conv2d42,
            constant111,
            conv2d43,
            constant113,
            conv2d44,
            constant115,
            constant116,
            constant117,
            constant118,
            conv2d45,
            constant120,
            conv2d46,
            constant122,
            conv2d47,
            constant124,
            constant125,
            constant126,
            constant127,
            conv2d48,
            constant129,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add51_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d38_out1 = self.conv2d38.forward(add51_out1);
        let constant100_out1 = self.constant100.val();
        let add52_out1 = conv2d38_out1.add(constant100_out1);
        let conv2d39_out1 = self.conv2d39.forward(add52_out1.clone());
        let constant102_out1 = self.constant102.val();
        let add53_out1 = conv2d39_out1.add(constant102_out1);
        let constant103_out1 = self.constant103.val();
        let div9_out1 = add53_out1
            .clone()
            .div((constant103_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf9_out1 = div9_out1.erf();
        let constant104_out1 = self.constant104.val();
        let add54_out1 = erf9_out1
            .add((constant104_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul21_out1 = add53_out1.mul(add54_out1);
        let constant105_out1 = self.constant105.val();
        let mul22_out1 = mul21_out1
            .mul((constant105_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d40_out1 = self.conv2d40.forward(mul22_out1);
        let constant107_out1 = self.constant107.val();
        let add55_out1 = conv2d40_out1.add(constant107_out1);
        let add56_out1 = add52_out1.add(add55_out1);
        let conv2d41_out1 = self.conv2d41.forward(add56_out1);
        let constant109_out1 = self.constant109.val();
        let add57_out1 = conv2d41_out1.add(constant109_out1);
        let reducemean5_out1 =
            { add57_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d42_out1 = self.conv2d42.forward(reducemean5_out1);
        let constant111_out1 = self.constant111.val();
        let add58_out1 = conv2d42_out1.add(constant111_out1);
        let relu10_out1 = burn::tensor::activation::relu(add58_out1);
        let conv2d43_out1 = self.conv2d43.forward(relu10_out1);
        let constant113_out1 = self.constant113.val();
        let add59_out1 = conv2d43_out1.add(constant113_out1);
        let hardsigmoid5_out1 = burn::tensor::activation::hard_sigmoid(
            add59_out1,
            0.16666670143604279,
            0.5,
        );
        let mul23_out1 = add57_out1.mul(hardsigmoid5_out1);
        let conv2d44_out1 = self.conv2d44.forward(mul23_out1.clone());
        let constant115_out1 = self.constant115.val();
        let add60_out1 = conv2d44_out1.add(constant115_out1);
        let constant116_out1 = self.constant116.val();
        let div10_out1 = add60_out1
            .clone()
            .div((constant116_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf10_out1 = div10_out1.erf();
        let constant117_out1 = self.constant117.val();
        let add61_out1 = erf10_out1
            .add((constant117_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul24_out1 = add60_out1.mul(add61_out1);
        let constant118_out1 = self.constant118.val();
        let mul25_out1 = mul24_out1
            .mul((constant118_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d45_out1 = self.conv2d45.forward(mul25_out1);
        let constant120_out1 = self.constant120.val();
        let add62_out1 = conv2d45_out1.add(constant120_out1);
        let add63_out1 = mul23_out1.add(add62_out1);
        let conv2d46_out1 = self.conv2d46.forward(add63_out1);
        let constant122_out1 = self.constant122.val();
        let add64_out1 = conv2d46_out1.add(constant122_out1);
        let conv2d47_out1 = self.conv2d47.forward(add64_out1.clone());
        let constant124_out1 = self.constant124.val();
        let add65_out1 = conv2d47_out1.add(constant124_out1);
        let constant125_out1 = self.constant125.val();
        let div11_out1 = add65_out1
            .clone()
            .div((constant125_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf11_out1 = div11_out1.erf();
        let constant126_out1 = self.constant126.val();
        let add66_out1 = erf11_out1
            .add((constant126_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul26_out1 = add65_out1.mul(add66_out1);
        let constant127_out1 = self.constant127.val();
        let mul27_out1 = mul26_out1
            .mul((constant127_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d48_out1 = self.conv2d48.forward(mul27_out1);
        let constant129_out1 = self.constant129.val();
        let add67_out1 = conv2d48_out1.add(constant129_out1);
        let add68_out1 = add64_out1.add(add67_out1);
        add68_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule5<B: Backend> {
    conv2d49: Conv2d<B>,
    constant131: burn::module::Param<Tensor<B, 4>>,
    conv2d50: Conv2d<B>,
    constant133: burn::module::Param<Tensor<B, 4>>,
    constant134: burn::module::Param<Tensor<B, 1>>,
    constant135: burn::module::Param<Tensor<B, 1>>,
    constant136: burn::module::Param<Tensor<B, 1>>,
    conv2d51: Conv2d<B>,
    constant138: burn::module::Param<Tensor<B, 4>>,
    conv2d52: Conv2d<B>,
    constant140: burn::module::Param<Tensor<B, 4>>,
    conv2d53: Conv2d<B>,
    constant142: burn::module::Param<Tensor<B, 4>>,
    conv2d54: Conv2d<B>,
    constant144: burn::module::Param<Tensor<B, 4>>,
    conv2d55: Conv2d<B>,
    constant146: burn::module::Param<Tensor<B, 4>>,
    constant147: burn::module::Param<Tensor<B, 1>>,
    constant148: burn::module::Param<Tensor<B, 1>>,
    constant149: burn::module::Param<Tensor<B, 1>>,
    conv2d56: Conv2d<B>,
    constant151: burn::module::Param<Tensor<B, 4>>,
    conv2d57: Conv2d<B>,
    constant153: burn::module::Param<Tensor<B, 4>>,
    conv2d58: Conv2d<B>,
    constant155: burn::module::Param<Tensor<B, 4>>,
    constant156: burn::module::Param<Tensor<B, 1>>,
    constant157: burn::module::Param<Tensor<B, 1>>,
    constant158: burn::module::Param<Tensor<B, 1>>,
    conv2d59: Conv2d<B>,
    constant160: burn::module::Param<Tensor<B, 4>>,
    averagepool2d1: AvgPool2d,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule5<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d49 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(false)
            .init(device);
        let constant131: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 512, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 512, 1, 1].into(),
            );
        let conv2d50 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant133: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1024, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1024, 1, 1].into(),
            );
        let constant134: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant135: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant136: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d51 = Conv2dConfig::new([1024, 768], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant138: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let conv2d52 = Conv2dConfig::new([768, 768], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(768)
            .with_bias(false)
            .init(device);
        let constant140: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let conv2d53 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant142: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 192, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 192, 1, 1].into(),
            );
        let conv2d54 = Conv2dConfig::new([192, 768], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant144: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let conv2d55 = Conv2dConfig::new([768, 1536], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant146: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1536, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1536, 1, 1].into(),
            );
        let constant147: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant148: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant149: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d56 = Conv2dConfig::new([1536, 768], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant151: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let conv2d57 = Conv2dConfig::new([768, 768], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(768)
            .with_bias(false)
            .init(device);
        let constant153: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let conv2d58 = Conv2dConfig::new([768, 1536], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant155: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 1536, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 1536, 1, 1].into(),
            );
        let constant156: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1.4142135381698608f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant157: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([1f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant158: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([0.5f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let conv2d59 = Conv2dConfig::new([1536, 768], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant160: burn::module::Param<Tensor<B, 4>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 4>::zeros(
                        [1, 768, 1, 1],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1, 768, 1, 1].into(),
            );
        let averagepool2d1 = AvgPool2dConfig::new([3, 2])
            .with_strides([3, 2])
            .with_padding(PaddingConfig2d::Valid)
            .with_count_include_pad(false)
            .with_ceil_mode(false)
            .init();
        Self {
            conv2d49,
            constant131,
            conv2d50,
            constant133,
            constant134,
            constant135,
            constant136,
            conv2d51,
            constant138,
            conv2d52,
            constant140,
            conv2d53,
            constant142,
            conv2d54,
            constant144,
            conv2d55,
            constant146,
            constant147,
            constant148,
            constant149,
            conv2d56,
            constant151,
            conv2d57,
            constant153,
            conv2d58,
            constant155,
            constant156,
            constant157,
            constant158,
            conv2d59,
            constant160,
            averagepool2d1,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, add68_out1: Tensor<B, 4>) -> Tensor<B, 4> {
        let conv2d49_out1 = self.conv2d49.forward(add68_out1);
        let constant131_out1 = self.constant131.val();
        let add69_out1 = conv2d49_out1.add(constant131_out1);
        let conv2d50_out1 = self.conv2d50.forward(add69_out1);
        let constant133_out1 = self.constant133.val();
        let add70_out1 = conv2d50_out1.add(constant133_out1);
        let constant134_out1 = self.constant134.val();
        let div12_out1 = add70_out1
            .clone()
            .div((constant134_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf12_out1 = div12_out1.erf();
        let constant135_out1 = self.constant135.val();
        let add71_out1 = erf12_out1
            .add((constant135_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul28_out1 = add70_out1.mul(add71_out1);
        let constant136_out1 = self.constant136.val();
        let mul29_out1 = mul28_out1
            .mul((constant136_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d51_out1 = self.conv2d51.forward(mul29_out1);
        let constant138_out1 = self.constant138.val();
        let add72_out1 = conv2d51_out1.add(constant138_out1);
        let conv2d52_out1 = self.conv2d52.forward(add72_out1);
        let constant140_out1 = self.constant140.val();
        let add73_out1 = conv2d52_out1.add(constant140_out1);
        let reducemean6_out1 =
            { add73_out1.clone().mean_dim(2usize).mean_dim(3usize) };
        let conv2d53_out1 = self.conv2d53.forward(reducemean6_out1);
        let constant142_out1 = self.constant142.val();
        let add74_out1 = conv2d53_out1.add(constant142_out1);
        let relu11_out1 = burn::tensor::activation::relu(add74_out1);
        let conv2d54_out1 = self.conv2d54.forward(relu11_out1);
        let constant144_out1 = self.constant144.val();
        let add75_out1 = conv2d54_out1.add(constant144_out1);
        let hardsigmoid6_out1 = burn::tensor::activation::hard_sigmoid(
            add75_out1,
            0.16666670143604279,
            0.5,
        );
        let mul30_out1 = add73_out1.mul(hardsigmoid6_out1);
        let conv2d55_out1 = self.conv2d55.forward(mul30_out1.clone());
        let constant146_out1 = self.constant146.val();
        let add76_out1 = conv2d55_out1.add(constant146_out1);
        let constant147_out1 = self.constant147.val();
        let div13_out1 = add76_out1
            .clone()
            .div((constant147_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf13_out1 = div13_out1.erf();
        let constant148_out1 = self.constant148.val();
        let add77_out1 = erf13_out1
            .add((constant148_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul31_out1 = add76_out1.mul(add77_out1);
        let constant149_out1 = self.constant149.val();
        let mul32_out1 = mul31_out1
            .mul((constant149_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d56_out1 = self.conv2d56.forward(mul32_out1);
        let constant151_out1 = self.constant151.val();
        let add78_out1 = conv2d56_out1.add(constant151_out1);
        let add79_out1 = mul30_out1.add(add78_out1);
        let conv2d57_out1 = self.conv2d57.forward(add79_out1);
        let constant153_out1 = self.constant153.val();
        let add80_out1 = conv2d57_out1.add(constant153_out1);
        let conv2d58_out1 = self.conv2d58.forward(add80_out1.clone());
        let constant155_out1 = self.constant155.val();
        let add81_out1 = conv2d58_out1.add(constant155_out1);
        let constant156_out1 = self.constant156.val();
        let div14_out1 = add81_out1
            .clone()
            .div((constant156_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf14_out1 = div14_out1.erf();
        let constant157_out1 = self.constant157.val();
        let add82_out1 = erf14_out1
            .add((constant157_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul33_out1 = add81_out1.mul(add82_out1);
        let constant158_out1 = self.constant158.val();
        let mul34_out1 = mul33_out1
            .mul((constant158_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d59_out1 = self.conv2d59.forward(mul34_out1);
        let constant160_out1 = self.constant160.val();
        let add83_out1 = conv2d59_out1.add(constant160_out1);
        let add84_out1 = add80_out1.add(add83_out1);
        let averagepool2d1_out1 = self.averagepool2d1.forward(add84_out1);
        averagepool2d1_out1
    }
}
#[derive(Module, Debug)]
pub struct Submodule6<B: Backend> {
    conv2d60: Conv2d<B>,
    batchnormalization1: BatchNorm<B>,
    conv2d61: Conv2d<B>,
    batchnormalization2: BatchNorm<B>,
    conv2d62: Conv2d<B>,
    batchnormalization3: BatchNorm<B>,
    concat2: burn::module::Param<Tensor<B, 1, Int>>,
    constant184: burn::module::Param<Tensor<B, 1>>,
    constant185: burn::module::Param<Tensor<B, 1>>,
    constant186: burn::module::Param<Tensor<B, 1>>,
    constant187: burn::module::Param<Tensor<B, 1>>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    constant207: burn::module::Param<Tensor<B, 1>>,
    constant208: burn::module::Param<Tensor<B, 1>>,
    constant209: burn::module::Param<Tensor<B, 1>>,
    constant210: burn::module::Param<Tensor<B, 1>>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    constant215: burn::module::Param<Tensor<B, 1>>,
    constant216: burn::module::Param<Tensor<B, 1>>,
    constant217: burn::module::Param<Tensor<B, 1>>,
    constant218: burn::module::Param<Tensor<B, 1>>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    constant238: burn::module::Param<Tensor<B, 1>>,
    constant239: burn::module::Param<Tensor<B, 1>>,
    constant240: burn::module::Param<Tensor<B, 1>>,
    constant241: burn::module::Param<Tensor<B, 1>>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    constant246: burn::module::Param<Tensor<B, 1>>,
    constant247: burn::module::Param<Tensor<B, 1>>,
    constant248: burn::module::Param<Tensor<B, 1>>,
    constant249: burn::module::Param<Tensor<B, 1>>,
    concat3: burn::module::Param<Tensor<B, 1, Int>>,
    linear9: Linear<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule6<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d60 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization1 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d61 = Conv2dConfig::new([768, 192], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let batchnormalization2 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let conv2d62 = Conv2dConfig::new([192, 192], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(192)
            .with_bias(false)
            .init(device);
        let batchnormalization3 = BatchNormConfig::new(192)
            .with_epsilon(0.000009999999747378752f64)
            .with_momentum(0.8999999761581421f64)
            .init(device);
        let concat2: burn::module::Param<Tensor<B, 1, Int>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1, Int>::zeros(
                        [3],
                        (device, burn::tensor::DType::I64),
                    )
                },
                device.clone(),
                false,
                [3].into(),
            );
        let constant184: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([2f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant185: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([
                            0.000009999999747378752f64,
                        ]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant186: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let constant187: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let linear1 = LinearConfig::new(192, 576).with_bias(true).init(device);
        let linear2 = LinearConfig::new(192, 192).with_bias(true).init(device);
        let constant207: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([2f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant208: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([
                            0.000009999999747378752f64,
                        ]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant209: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let constant210: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let linear3 = LinearConfig::new(192, 768).with_bias(true).init(device);
        let linear4 = LinearConfig::new(768, 192).with_bias(true).init(device);
        let constant215: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([2f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant216: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([
                            0.000009999999747378752f64,
                        ]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant217: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let constant218: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let linear5 = LinearConfig::new(192, 576).with_bias(true).init(device);
        let linear6 = LinearConfig::new(192, 192).with_bias(true).init(device);
        let constant238: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([2f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant239: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([
                            0.000009999999747378752f64,
                        ]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant240: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let constant241: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let linear7 = LinearConfig::new(192, 768).with_bias(true).init(device);
        let linear8 = LinearConfig::new(768, 192).with_bias(true).init(device);
        let constant246: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([2f64]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant247: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::from_data(
                        burn::tensor::TensorData::from([
                            0.0000009999999974752427f64,
                        ]),
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [1].into(),
            );
        let constant248: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let constant249: burn::module::Param<Tensor<B, 1>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1>::zeros(
                        [192],
                        (device, burn::tensor::DType::F32),
                    )
                },
                device.clone(),
                false,
                [192].into(),
            );
        let concat3: burn::module::Param<Tensor<B, 1, Int>> =
            burn::module::Param::uninitialized(
                burn::module::ParamId::new(),
                move |device, _require_grad| {
                    Tensor::<B, 1, Int>::zeros(
                        [4],
                        (device, burn::tensor::DType::I64),
                    )
                },
                device.clone(),
                false,
                [4].into(),
            );
        let linear9 =
            LinearConfig::new(192, 18710).with_bias(true).init(device);
        Self {
            conv2d60,
            batchnormalization1,
            conv2d61,
            batchnormalization2,
            conv2d62,
            batchnormalization3,
            concat2,
            constant184,
            constant185,
            constant186,
            constant187,
            linear1,
            linear2,
            constant207,
            constant208,
            constant209,
            constant210,
            linear3,
            linear4,
            constant215,
            constant216,
            constant217,
            constant218,
            linear5,
            linear6,
            constant238,
            constant239,
            constant240,
            constant241,
            linear7,
            linear8,
            constant246,
            constant247,
            constant248,
            constant249,
            concat3,
            linear9,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, averagepool2d1_out1: Tensor<B, 4>) -> Tensor<B, 3> {
        let conv2d60_out1 = self.conv2d60.forward(averagepool2d1_out1.clone());
        let batchnormalization1_out1 =
            self.batchnormalization1.forward(conv2d60_out1);
        let sigmoid1_out1 =
            burn::tensor::activation::sigmoid(batchnormalization1_out1.clone());
        let mul35_out1 = batchnormalization1_out1.mul(sigmoid1_out1);
        let conv2d61_out1 = self.conv2d61.forward(averagepool2d1_out1);
        let batchnormalization2_out1 =
            self.batchnormalization2.forward(conv2d61_out1);
        let sigmoid2_out1 =
            burn::tensor::activation::sigmoid(batchnormalization2_out1.clone());
        let mul36_out1 = batchnormalization2_out1.mul(sigmoid2_out1);
        let conv2d62_out1 = self.conv2d62.forward(mul36_out1.clone());
        let batchnormalization3_out1 =
            self.batchnormalization3.forward(conv2d62_out1);
        let sigmoid3_out1 =
            burn::tensor::activation::sigmoid(batchnormalization3_out1.clone());
        let mul37_out1 = batchnormalization3_out1.mul(sigmoid3_out1);
        let add85_out1 = mul36_out1.add(mul37_out1);
        let concat2_out1 = self.concat2.val();
        let reshape1_out1 = add85_out1.reshape([1, 192, -1]);
        let transpose1_out1 = reshape1_out1.permute([0, 2, 1]);
        let reducemean7_out1 = { transpose1_out1.clone().mean_dim(2usize) };
        let sub1_out1 = transpose1_out1.clone().sub(reducemean7_out1);
        let constant184_out1 = self.constant184.val();
        let pow1_out1 = sub1_out1
            .clone()
            .powf((constant184_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean8_out1 = { pow1_out1.mean_dim(2usize) };
        let constant185_out1 = self.constant185.val();
        let add86_out1 = reducemean8_out1
            .add((constant185_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt1_out1 = add86_out1.sqrt();
        let div15_out1 = sub1_out1.div(sqrt1_out1);
        let constant186_out1 = self.constant186.val();
        let mul38_out1 = div15_out1
            .mul((constant186_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant187_out1 = self.constant187.val();
        let add87_out1 = mul38_out1
            .add((constant187_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear1_out1 = self.linear1.forward(add87_out1);
        let reshape2_out1 = linear1_out1.reshape([0, -1, 3, 8, 24]);
        let transpose2_out1 = reshape2_out1.permute([2, 0, 3, 1, 4]);
        let slice3_out1 =
            transpose2_out1.clone().slice(s![0..1, .., .., .., ..]);
        let squeeze2_out1 = slice3_out1.squeeze_dims::<4>(&[0]);
        let slice4_out1 =
            transpose2_out1.clone().slice(s![1..2, .., .., .., ..]);
        let squeeze3_out1 = slice4_out1.squeeze_dims::<4>(&[0]);
        let slice5_out1 = transpose2_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze4_out1 = slice5_out1.squeeze_dims::<4>(&[0]);
        let (matmul3_out1,) = {
            let q = squeeze2_out1;
            let k = squeeze3_out1;
            let v = squeeze4_out1;
            let matmul3_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.20412415266036987f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul3_out1,)
        };
        let transpose4_out1 = matmul3_out1.permute([0, 2, 1, 3]);
        let reshape4_out1 = transpose4_out1.reshape([0, -1, 192]);
        let linear2_out1 = self.linear2.forward(reshape4_out1);
        let add88_out1 = transpose1_out1.add(linear2_out1);
        let reducemean9_out1 = { add88_out1.clone().mean_dim(2usize) };
        let sub2_out1 = add88_out1.clone().sub(reducemean9_out1);
        let constant207_out1 = self.constant207.val();
        let pow2_out1 = sub2_out1
            .clone()
            .powf((constant207_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean10_out1 = { pow2_out1.mean_dim(2usize) };
        let constant208_out1 = self.constant208.val();
        let add89_out1 = reducemean10_out1
            .add((constant208_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt2_out1 = add89_out1.sqrt();
        let div16_out1 = sub2_out1.div(sqrt2_out1);
        let constant209_out1 = self.constant209.val();
        let mul40_out1 = div16_out1
            .mul((constant209_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant210_out1 = self.constant210.val();
        let add90_out1 = mul40_out1
            .add((constant210_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear3_out1 = self.linear3.forward(add90_out1);
        let sigmoid4_out1 =
            burn::tensor::activation::sigmoid(linear3_out1.clone());
        let mul41_out1 = linear3_out1.mul(sigmoid4_out1);
        let linear4_out1 = self.linear4.forward(mul41_out1);
        let add91_out1 = add88_out1.add(linear4_out1);
        let reducemean11_out1 = { add91_out1.clone().mean_dim(2usize) };
        let sub3_out1 = add91_out1.clone().sub(reducemean11_out1);
        let constant215_out1 = self.constant215.val();
        let pow3_out1 = sub3_out1
            .clone()
            .powf((constant215_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean12_out1 = { pow3_out1.mean_dim(2usize) };
        let constant216_out1 = self.constant216.val();
        let add92_out1 = reducemean12_out1
            .add((constant216_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt3_out1 = add92_out1.sqrt();
        let div17_out1 = sub3_out1.div(sqrt3_out1);
        let constant217_out1 = self.constant217.val();
        let mul42_out1 = div17_out1
            .mul((constant217_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant218_out1 = self.constant218.val();
        let add93_out1 = mul42_out1
            .add((constant218_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear5_out1 = self.linear5.forward(add93_out1);
        let reshape5_out1 = linear5_out1.reshape([0, -1, 3, 8, 24]);
        let transpose5_out1 = reshape5_out1.permute([2, 0, 3, 1, 4]);
        let slice6_out1 =
            transpose5_out1.clone().slice(s![0..1, .., .., .., ..]);
        let squeeze5_out1 = slice6_out1.squeeze_dims::<4>(&[0]);
        let slice7_out1 =
            transpose5_out1.clone().slice(s![1..2, .., .., .., ..]);
        let squeeze6_out1 = slice7_out1.squeeze_dims::<4>(&[0]);
        let slice8_out1 = transpose5_out1.slice(s![2..3, .., .., .., ..]);
        let squeeze7_out1 = slice8_out1.squeeze_dims::<4>(&[0]);
        let (matmul9_out1,) = {
            let q = squeeze5_out1;
            let k = squeeze6_out1;
            let v = squeeze7_out1;
            let matmul9_out1 = burn::tensor::module::attention(
                q,
                k,
                v,
                None,
                None,
                burn::tensor::ops::AttentionModuleOptions {
                    scale: Some(0.20412415266036987f64),
                    softcap: None,
                    is_causal: false,
                },
            );
            (matmul9_out1,)
        };
        let transpose7_out1 = matmul9_out1.permute([0, 2, 1, 3]);
        let reshape7_out1 = transpose7_out1.reshape([0, -1, 192]);
        let linear6_out1 = self.linear6.forward(reshape7_out1);
        let add94_out1 = add91_out1.add(linear6_out1);
        let reducemean13_out1 = { add94_out1.clone().mean_dim(2usize) };
        let sub4_out1 = add94_out1.clone().sub(reducemean13_out1);
        let constant238_out1 = self.constant238.val();
        let pow4_out1 = sub4_out1
            .clone()
            .powf((constant238_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean14_out1 = { pow4_out1.mean_dim(2usize) };
        let constant239_out1 = self.constant239.val();
        let add95_out1 = reducemean14_out1
            .add((constant239_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt4_out1 = add95_out1.sqrt();
        let div18_out1 = sub4_out1.div(sqrt4_out1);
        let constant240_out1 = self.constant240.val();
        let mul44_out1 = div18_out1
            .mul((constant240_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant241_out1 = self.constant241.val();
        let add96_out1 = mul44_out1
            .add((constant241_out1).unsqueeze_dims(&[0isize, 1isize]));
        let linear7_out1 = self.linear7.forward(add96_out1);
        let sigmoid5_out1 =
            burn::tensor::activation::sigmoid(linear7_out1.clone());
        let mul45_out1 = linear7_out1.mul(sigmoid5_out1);
        let linear8_out1 = self.linear8.forward(mul45_out1);
        let add97_out1 = add94_out1.add(linear8_out1);
        let reducemean15_out1 = { add97_out1.clone().mean_dim(2usize) };
        let sub5_out1 = add97_out1.sub(reducemean15_out1);
        let constant246_out1 = self.constant246.val();
        let pow5_out1 = sub5_out1
            .clone()
            .powf((constant246_out1).unsqueeze_dims(&[0isize, 1isize]));
        let reducemean16_out1 = { pow5_out1.mean_dim(2usize) };
        let constant247_out1 = self.constant247.val();
        let add98_out1 = reducemean16_out1
            .add((constant247_out1).unsqueeze_dims(&[0isize, 1isize]));
        let sqrt5_out1 = add98_out1.sqrt();
        let div19_out1 = sub5_out1.div(sqrt5_out1);
        let constant248_out1 = self.constant248.val();
        let mul46_out1 = div19_out1
            .mul((constant248_out1).unsqueeze_dims(&[0isize, 1isize]));
        let constant249_out1 = self.constant249.val();
        let add99_out1 = mul46_out1
            .add((constant249_out1).unsqueeze_dims(&[0isize, 1isize]));
        let concat3_out1 = self.concat3.val();
        let reshape8_out1 = add99_out1.reshape([0, 1, 40, 192]);
        let transpose8_out1 = reshape8_out1.permute([0, 3, 1, 2]);
        let add100_out1 = transpose8_out1.add(mul35_out1);
        let squeeze8_out1 = add100_out1.squeeze_dims::<3>(&[2]);
        let transpose9_out1 = squeeze8_out1.permute([0, 2, 1]);
        let linear9_out1 = self.linear9.forward(transpose9_out1);
        let softmax3_out1 = burn::tensor::activation::softmax(linear9_out1, 2);
        softmax3_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    submodule3: Submodule3<B>,
    submodule4: Submodule4<B>,
    submodule5: Submodule5<B>,
    submodule6: Submodule6<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}

impl<B: Backend> Model<B> {
    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model
            .load_from(&mut store)
            .expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        let submodule3 = Submodule3::new(device);
        let submodule4 = Submodule4::new(device);
        let submodule5 = Submodule5::new(device);
        let submodule6 = Submodule6::new(device);
        Self {
            submodule1,
            submodule2,
            submodule3,
            submodule4,
            submodule5,
            submodule6,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let mul5_out1 = self.submodule1.forward(x);
        let conv2d24_out1 = self.submodule2.forward(mul5_out1);
        let add51_out1 = self.submodule3.forward(conv2d24_out1);
        let add68_out1 = self.submodule4.forward(add51_out1);
        let averagepool2d1_out1 = self.submodule5.forward(add68_out1);
        let softmax3_out1 = self.submodule6.forward(averagepool2d1_out1);
        softmax3_out1
    }
}
