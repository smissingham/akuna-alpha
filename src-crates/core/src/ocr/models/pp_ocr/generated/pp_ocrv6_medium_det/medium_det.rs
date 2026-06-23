#![allow(clippy::unwrap_used, clippy::unnecessary_cast)]
// Generated from ONNX "/var/folders/r8/v0v7bpr93jn7636n86zxwn0r0000gn/T/nix-shell.kwhRLa/akuna-pp-ocr-codegen/patched/medium_det.onnx" by burn-onnx
use burn::prelude::*;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::Conv2d;
use burn::nn::conv::Conv2dConfig;
use burn::nn::conv::ConvTranspose2d;
use burn::nn::conv::ConvTranspose2dConfig;
use burn::nn::pool::MaxPool2d;
use burn::nn::pool::MaxPool2dConfig;
use burn::tensor::Bytes;
use burn_store::BurnpackStore;
use burn_store::ModuleSnapshot;


#[derive(Module, Debug)]
pub struct Submodule1<B: Backend> {
    conv2d1: Conv2d<B>,
    conv2d2: Conv2d<B>,
    conv2d3: Conv2d<B>,
    maxpool2d1: MaxPool2d,
    conv2d4: Conv2d<B>,
    conv2d5: Conv2d<B>,
    conv2d6: Conv2d<B>,
    conv2d7: Conv2d<B>,
    conv2d8: Conv2d<B>,
    conv2d9: Conv2d<B>,
    constant10: burn::module::Param<Tensor<B, 1>>,
    constant11: burn::module::Param<Tensor<B, 1>>,
    constant12: burn::module::Param<Tensor<B, 1>>,
    conv2d10: Conv2d<B>,
    conv2d11: Conv2d<B>,
    conv2d12: Conv2d<B>,
    conv2d13: Conv2d<B>,
    conv2d14: Conv2d<B>,
    conv2d15: Conv2d<B>,
    conv2d16: Conv2d<B>,
    conv2d17: Conv2d<B>,
    conv2d18: Conv2d<B>,
    conv2d19: Conv2d<B>,
    conv2d20: Conv2d<B>,
    conv2d21: Conv2d<B>,
    conv2d22: Conv2d<B>,
    conv2d23: Conv2d<B>,
    conv2d24: Conv2d<B>,
    conv2d25: Conv2d<B>,
    conv2d26: Conv2d<B>,
    conv2d27: Conv2d<B>,
    conv2d28: Conv2d<B>,
    conv2d29: Conv2d<B>,
    conv2d30: Conv2d<B>,
    conv2d31: Conv2d<B>,
    conv2d32: Conv2d<B>,
    conv2d33: Conv2d<B>,
    conv2d34: Conv2d<B>,
    conv2d35: Conv2d<B>,
    conv2d36: Conv2d<B>,
    conv2d37: Conv2d<B>,
    conv2d38: Conv2d<B>,
    conv2d39: Conv2d<B>,
    conv2d40: Conv2d<B>,
    conv2d41: Conv2d<B>,
    conv2d42: Conv2d<B>,
    conv2d43: Conv2d<B>,
    conv2d44: Conv2d<B>,
    conv2d45: Conv2d<B>,
    conv2d46: Conv2d<B>,
    conv2d47: Conv2d<B>,
    conv2d48: Conv2d<B>,
    conv2d49: Conv2d<B>,
    conv2d50: Conv2d<B>,
    conv2d51: Conv2d<B>,
    conv2d52: Conv2d<B>,
    conv2d53: Conv2d<B>,
    conv2d54: Conv2d<B>,
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
            .with_bias(true)
            .init(device);
        let conv2d2 = Conv2dConfig::new([64, 32], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d3 = Conv2dConfig::new([32, 64], [2, 2])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
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
            .with_bias(true)
            .init(device);
        let conv2d5 = Conv2dConfig::new([64, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d6 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d7 = Conv2dConfig::new([128, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d8 = Conv2dConfig::new([32, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d9 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let constant10: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<
                B,
                1,
            >::from_data([1.4142135381698608f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant11: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([1f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let constant12: burn::module::Param<Tensor<B, 1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 1>::from_data([0.5f64], device),
            device.clone(),
            false,
            [1].into(),
        );
        let conv2d10 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d11 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d12 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d13 = Conv2dConfig::new([256, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d14 = Conv2dConfig::new([128, 128], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(128)
            .with_bias(true)
            .init(device);
        let conv2d15 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d16 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d17 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d18 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d19 = Conv2dConfig::new([64, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d20 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d21 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d22 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d23 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d24 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d25 = Conv2dConfig::new([256, 256], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d26 = Conv2dConfig::new([256, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d27 = Conv2dConfig::new([512, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d28 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d29 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d30 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d31 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d32 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d33 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d34 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d35 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d36 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d37 = Conv2dConfig::new([512, 128], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d38 = Conv2dConfig::new([128, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d39 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d40 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d41 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d42 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d43 = Conv2dConfig::new([1024, 512], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d44 = Conv2dConfig::new([512, 512], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(512)
            .with_bias(true)
            .init(device);
        let conv2d45 = Conv2dConfig::new([512, 1024], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d46 = Conv2dConfig::new([1024, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d47 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(896)
            .with_bias(true)
            .init(device);
        let conv2d48 = Conv2dConfig::new([896, 224], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d49 = Conv2dConfig::new([224, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d50 = Conv2dConfig::new([896, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d51 = Conv2dConfig::new([1792, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d52 = Conv2dConfig::new([896, 896], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(896)
            .with_bias(true)
            .init(device);
        let conv2d53 = Conv2dConfig::new([896, 1792], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d54 = Conv2dConfig::new([1792, 896], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        Self {
            conv2d1,
            conv2d2,
            conv2d3,
            maxpool2d1,
            conv2d4,
            conv2d5,
            conv2d6,
            conv2d7,
            conv2d8,
            conv2d9,
            constant10,
            constant11,
            constant12,
            conv2d10,
            conv2d11,
            conv2d12,
            conv2d13,
            conv2d14,
            conv2d15,
            conv2d16,
            conv2d17,
            conv2d18,
            conv2d19,
            conv2d20,
            conv2d21,
            conv2d22,
            conv2d23,
            conv2d24,
            conv2d25,
            conv2d26,
            conv2d27,
            conv2d28,
            conv2d29,
            conv2d30,
            conv2d31,
            conv2d32,
            conv2d33,
            conv2d34,
            conv2d35,
            conv2d36,
            conv2d37,
            conv2d38,
            conv2d39,
            conv2d40,
            conv2d41,
            conv2d42,
            conv2d43,
            conv2d44,
            conv2d45,
            conv2d46,
            conv2d47,
            conv2d48,
            conv2d49,
            conv2d50,
            conv2d51,
            conv2d52,
            conv2d53,
            conv2d54,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let conv2d1_out1 = self.conv2d1.forward(x);
        let relu1_out1 = burn::tensor::activation::relu(conv2d1_out1);
        let conv2d2_out1 = self.conv2d2.forward(relu1_out1.clone());
        let relu2_out1 = burn::tensor::activation::relu(conv2d2_out1);
        let conv2d3_out1 = self.conv2d3.forward(relu2_out1);
        let relu3_out1 = burn::tensor::activation::relu(conv2d3_out1);
        let maxpool2d1_out1 = self.maxpool2d1.forward(relu1_out1);
        let concat1_out1 = burn::tensor::Tensor::cat(
            [maxpool2d1_out1, relu3_out1].into(),
            1,
        );
        let conv2d4_out1 = self.conv2d4.forward(concat1_out1);
        let relu4_out1 = burn::tensor::activation::relu(conv2d4_out1);
        let conv2d5_out1 = self.conv2d5.forward(relu4_out1);
        let relu5_out1 = burn::tensor::activation::relu(conv2d5_out1);
        let conv2d6_out1 = self.conv2d6.forward(relu5_out1);
        let reducemean1_out1 = {
            conv2d6_out1.clone().mean_dim(2usize).mean_dim(3usize)
        };
        let conv2d7_out1 = self.conv2d7.forward(reducemean1_out1);
        let relu6_out1 = burn::tensor::activation::relu(conv2d7_out1);
        let conv2d8_out1 = self.conv2d8.forward(relu6_out1);
        let hardsigmoid1_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d8_out1,
            0.16666670143604279,
            0.5,
        );
        let mul1_out1 = conv2d6_out1.mul(hardsigmoid1_out1);
        let conv2d9_out1 = self.conv2d9.forward(mul1_out1.clone());
        let constant10_out1 = self.constant10.val();
        let div1_out1 = conv2d9_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf1_out1 = div1_out1.erf();
        let constant11_out1 = self.constant11.val();
        let add1_out1 = erf1_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul2_out1 = conv2d9_out1.mul(add1_out1);
        let constant12_out1 = self.constant12.val();
        let mul3_out1 = mul2_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d10_out1 = self.conv2d10.forward(mul3_out1);
        let add2_out1 = mul1_out1.add(conv2d10_out1);
        let conv2d11_out1 = self.conv2d11.forward(add2_out1);
        let conv2d12_out1 = self.conv2d12.forward(conv2d11_out1.clone());
        let div2_out1 = conv2d12_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf2_out1 = div2_out1.erf();
        let add3_out1 = erf2_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul4_out1 = conv2d12_out1.mul(add3_out1);
        let mul5_out1 = mul4_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d13_out1 = self.conv2d13.forward(mul5_out1);
        let add4_out1 = conv2d11_out1.add(conv2d13_out1);
        let conv2d14_out1 = self.conv2d14.forward(add4_out1.clone());
        let conv2d15_out1 = self.conv2d15.forward(conv2d14_out1);
        let div3_out1 = conv2d15_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf3_out1 = div3_out1.erf();
        let add5_out1 = erf3_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul6_out1 = conv2d15_out1.mul(add5_out1);
        let mul7_out1 = mul6_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d16_out1 = self.conv2d16.forward(mul7_out1);
        let conv2d17_out1 = self.conv2d17.forward(conv2d16_out1);
        let reducemean2_out1 = {
            conv2d17_out1.clone().mean_dim(2usize).mean_dim(3usize)
        };
        let conv2d18_out1 = self.conv2d18.forward(reducemean2_out1);
        let relu7_out1 = burn::tensor::activation::relu(conv2d18_out1);
        let conv2d19_out1 = self.conv2d19.forward(relu7_out1);
        let hardsigmoid2_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d19_out1,
            0.16666670143604279,
            0.5,
        );
        let mul8_out1 = conv2d17_out1.mul(hardsigmoid2_out1);
        let conv2d20_out1 = self.conv2d20.forward(mul8_out1.clone());
        let div4_out1 = conv2d20_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf4_out1 = div4_out1.erf();
        let add6_out1 = erf4_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul9_out1 = conv2d20_out1.mul(add6_out1);
        let mul10_out1 = mul9_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d21_out1 = self.conv2d21.forward(mul10_out1);
        let add7_out1 = mul8_out1.add(conv2d21_out1);
        let conv2d22_out1 = self.conv2d22.forward(add7_out1);
        let conv2d23_out1 = self.conv2d23.forward(conv2d22_out1.clone());
        let div5_out1 = conv2d23_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf5_out1 = div5_out1.erf();
        let add8_out1 = erf5_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul11_out1 = conv2d23_out1.mul(add8_out1);
        let mul12_out1 = mul11_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d24_out1 = self.conv2d24.forward(mul12_out1);
        let add9_out1 = conv2d22_out1.add(conv2d24_out1);
        let conv2d25_out1 = self.conv2d25.forward(add9_out1.clone());
        let conv2d26_out1 = self.conv2d26.forward(conv2d25_out1);
        let div6_out1 = conv2d26_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf6_out1 = div6_out1.erf();
        let add10_out1 = erf6_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul13_out1 = conv2d26_out1.mul(add10_out1);
        let mul14_out1 = mul13_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d27_out1 = self.conv2d27.forward(mul14_out1);
        let conv2d28_out1 = self.conv2d28.forward(conv2d27_out1);
        let reducemean3_out1 = {
            conv2d28_out1.clone().mean_dim(2usize).mean_dim(3usize)
        };
        let conv2d29_out1 = self.conv2d29.forward(reducemean3_out1);
        let relu8_out1 = burn::tensor::activation::relu(conv2d29_out1);
        let conv2d30_out1 = self.conv2d30.forward(relu8_out1);
        let hardsigmoid3_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d30_out1,
            0.16666670143604279,
            0.5,
        );
        let mul15_out1 = conv2d28_out1.mul(hardsigmoid3_out1);
        let conv2d31_out1 = self.conv2d31.forward(mul15_out1.clone());
        let div7_out1 = conv2d31_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf7_out1 = div7_out1.erf();
        let add11_out1 = erf7_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul16_out1 = conv2d31_out1.mul(add11_out1);
        let mul17_out1 = mul16_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d32_out1 = self.conv2d32.forward(mul17_out1);
        let add12_out1 = mul15_out1.add(conv2d32_out1);
        let conv2d33_out1 = self.conv2d33.forward(add12_out1);
        let conv2d34_out1 = self.conv2d34.forward(conv2d33_out1.clone());
        let div8_out1 = conv2d34_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf8_out1 = div8_out1.erf();
        let add13_out1 = erf8_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul18_out1 = conv2d34_out1.mul(add13_out1);
        let mul19_out1 = mul18_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d35_out1 = self.conv2d35.forward(mul19_out1);
        let add14_out1 = conv2d33_out1.add(conv2d35_out1);
        let conv2d36_out1 = self.conv2d36.forward(add14_out1);
        let reducemean4_out1 = {
            conv2d36_out1.clone().mean_dim(2usize).mean_dim(3usize)
        };
        let conv2d37_out1 = self.conv2d37.forward(reducemean4_out1);
        let relu9_out1 = burn::tensor::activation::relu(conv2d37_out1);
        let conv2d38_out1 = self.conv2d38.forward(relu9_out1);
        let hardsigmoid4_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d38_out1,
            0.16666670143604279,
            0.5,
        );
        let mul20_out1 = conv2d36_out1.mul(hardsigmoid4_out1);
        let conv2d39_out1 = self.conv2d39.forward(mul20_out1.clone());
        let div9_out1 = conv2d39_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf9_out1 = div9_out1.erf();
        let add15_out1 = erf9_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul21_out1 = conv2d39_out1.mul(add15_out1);
        let mul22_out1 = mul21_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d40_out1 = self.conv2d40.forward(mul22_out1);
        let add16_out1 = mul20_out1.add(conv2d40_out1);
        let conv2d41_out1 = self.conv2d41.forward(add16_out1);
        let conv2d42_out1 = self.conv2d42.forward(conv2d41_out1.clone());
        let div10_out1 = conv2d42_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf10_out1 = div10_out1.erf();
        let add17_out1 = erf10_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul23_out1 = conv2d42_out1.mul(add17_out1);
        let mul24_out1 = mul23_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d43_out1 = self.conv2d43.forward(mul24_out1);
        let add18_out1 = conv2d41_out1.add(conv2d43_out1);
        let conv2d44_out1 = self.conv2d44.forward(add18_out1.clone());
        let conv2d45_out1 = self.conv2d45.forward(conv2d44_out1);
        let div11_out1 = conv2d45_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf11_out1 = div11_out1.erf();
        let add19_out1 = erf11_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul25_out1 = conv2d45_out1.mul(add19_out1);
        let mul26_out1 = mul25_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d46_out1 = self.conv2d46.forward(mul26_out1);
        let conv2d47_out1 = self.conv2d47.forward(conv2d46_out1);
        let reducemean5_out1 = {
            conv2d47_out1.clone().mean_dim(2usize).mean_dim(3usize)
        };
        let conv2d48_out1 = self.conv2d48.forward(reducemean5_out1);
        let relu10_out1 = burn::tensor::activation::relu(conv2d48_out1);
        let conv2d49_out1 = self.conv2d49.forward(relu10_out1);
        let hardsigmoid5_out1 = burn::tensor::activation::hard_sigmoid(
            conv2d49_out1,
            0.16666670143604279,
            0.5,
        );
        let mul27_out1 = conv2d47_out1.mul(hardsigmoid5_out1);
        let conv2d50_out1 = self.conv2d50.forward(mul27_out1.clone());
        let div12_out1 = conv2d50_out1
            .clone()
            .div((constant10_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf12_out1 = div12_out1.erf();
        let add20_out1 = erf12_out1
            .add((constant11_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul28_out1 = conv2d50_out1.mul(add20_out1);
        let mul29_out1 = mul28_out1
            .mul((constant12_out1.clone()).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d51_out1 = self.conv2d51.forward(mul29_out1);
        let add21_out1 = mul27_out1.add(conv2d51_out1);
        let conv2d52_out1 = self.conv2d52.forward(add21_out1);
        let conv2d53_out1 = self.conv2d53.forward(conv2d52_out1.clone());
        let div13_out1 = conv2d53_out1
            .clone()
            .div((constant10_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let erf13_out1 = div13_out1.erf();
        let add22_out1 = erf13_out1
            .add((constant11_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let mul30_out1 = conv2d53_out1.mul(add22_out1);
        let mul31_out1 = mul30_out1
            .mul((constant12_out1).unsqueeze_dims(&[0isize, 1isize, 2isize]));
        let conv2d54_out1 = self.conv2d54.forward(mul31_out1);
        let add23_out1 = conv2d52_out1.add(conv2d54_out1);
        (add23_out1, add18_out1, add9_out1, add4_out1)
    }
}
#[derive(Module, Debug)]
pub struct Submodule2<B: Backend> {
    conv2d55: Conv2d<B>,
    conv2d56: Conv2d<B>,
    conv2d57: Conv2d<B>,
    conv2d58: Conv2d<B>,
    resize1: burn::nn::interpolate::Interpolate2d,
    resize2: burn::nn::interpolate::Interpolate2d,
    resize3: burn::nn::interpolate::Interpolate2d,
    conv2d59: Conv2d<B>,
    conv2d60: Conv2d<B>,
    conv2d61: Conv2d<B>,
    conv2d62: Conv2d<B>,
    conv2d63: Conv2d<B>,
    conv2d64: Conv2d<B>,
    conv2d65: Conv2d<B>,
    conv2d66: Conv2d<B>,
    conv2d67: Conv2d<B>,
    conv2d68: Conv2d<B>,
    conv2d69: Conv2d<B>,
    conv2d70: Conv2d<B>,
    conv2d71: Conv2d<B>,
    conv2d72: Conv2d<B>,
    conv2d73: Conv2d<B>,
    conv2d74: Conv2d<B>,
    conv2d75: Conv2d<B>,
    conv2d76: Conv2d<B>,
    conv2d77: Conv2d<B>,
    conv2d78: Conv2d<B>,
    conv2d79: Conv2d<B>,
    conv2d80: Conv2d<B>,
    conv2d81: Conv2d<B>,
    conv2d82: Conv2d<B>,
    conv2d83: Conv2d<B>,
    conv2d84: Conv2d<B>,
    conv2d85: Conv2d<B>,
    conv2d86: Conv2d<B>,
    conv2d87: Conv2d<B>,
    conv2d88: Conv2d<B>,
    conv2d89: Conv2d<B>,
    conv2d90: Conv2d<B>,
    conv2d91: Conv2d<B>,
    conv2d92: Conv2d<B>,
    conv2d93: Conv2d<B>,
    conv2d94: Conv2d<B>,
    conv2d95: Conv2d<B>,
    conv2d96: Conv2d<B>,
    conv2d97: Conv2d<B>,
    conv2d98: Conv2d<B>,
    conv2d99: Conv2d<B>,
    conv2d100: Conv2d<B>,
    conv2d101: Conv2d<B>,
    conv2d102: Conv2d<B>,
    conv2d103: Conv2d<B>,
    conv2d104: Conv2d<B>,
    conv2d105: Conv2d<B>,
    conv2d106: Conv2d<B>,
    conv2d107: Conv2d<B>,
    conv2d108: Conv2d<B>,
    conv2d109: Conv2d<B>,
    conv2d110: Conv2d<B>,
    conv2d111: Conv2d<B>,
    conv2d112: Conv2d<B>,
    conv2d113: Conv2d<B>,
    conv2d114: Conv2d<B>,
    conv2d115: Conv2d<B>,
    conv2d116: Conv2d<B>,
    conv2d117: Conv2d<B>,
    conv2d118: Conv2d<B>,
    conv2d119: Conv2d<B>,
    conv2d120: Conv2d<B>,
    conv2d121: Conv2d<B>,
    resize4: burn::nn::interpolate::Interpolate2d,
    resize5: burn::nn::interpolate::Interpolate2d,
    resize6: burn::nn::interpolate::Interpolate2d,
    conv2d122: Conv2d<B>,
    convtranspose2d1: ConvTranspose2d<B>,
    constant127: burn::module::Param<Tensor<B, 4>>,
    convtranspose2d2: ConvTranspose2d<B>,
    constant129: burn::module::Param<Tensor<B, 4>>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}
impl<B: Backend> Submodule2<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let conv2d55 = Conv2dConfig::new([896, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d56 = Conv2dConfig::new([512, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d57 = Conv2dConfig::new([256, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d58 = Conv2dConfig::new([128, 256], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let resize1 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize2 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize3 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d59 = Conv2dConfig::new([256, 256], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d60 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d61 = Conv2dConfig::new([256, 256], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d62 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d63 = Conv2dConfig::new([256, 256], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d64 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d65 = Conv2dConfig::new([256, 256], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(256)
            .with_bias(true)
            .init(device);
        let conv2d66 = Conv2dConfig::new([256, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d67 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d68 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d69 = Conv2dConfig::new([64, 64], [3, 3])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let conv2d70 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d71 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d72 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d73 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d74 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d75 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d76 = Conv2dConfig::new([64, 64], [9, 9])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(4, 4, 4, 4))
            .with_dilation([1, 1])
            .with_groups(64)
            .with_bias(true)
            .init(device);
        let conv2d77 = Conv2dConfig::new([64, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d78 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d79 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d80 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d81 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d82 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d83 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d84 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d85 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d86 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d87 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d88 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d89 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d90 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d91 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d92 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d93 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d94 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d95 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d96 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d97 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d98 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d99 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d100 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d101 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d102 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d103 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d104 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d105 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d106 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d107 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d108 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d109 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d110 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d111 = Conv2dConfig::new([64, 32], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d112 = Conv2dConfig::new([32, 32], [7, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d113 = Conv2dConfig::new([32, 32], [7, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(3, 0, 3, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d114 = Conv2dConfig::new([32, 32], [1, 7])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 3, 0, 3))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d115 = Conv2dConfig::new([32, 32], [5, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 2, 2, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d116 = Conv2dConfig::new([32, 32], [5, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(2, 0, 2, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d117 = Conv2dConfig::new([32, 32], [1, 5])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 2, 0, 2))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d118 = Conv2dConfig::new([32, 32], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d119 = Conv2dConfig::new([32, 32], [3, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 0, 1, 0))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d120 = Conv2dConfig::new([32, 32], [1, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 1, 0, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let conv2d121 = Conv2dConfig::new([32, 64], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let resize4 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([8.0, 8.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize5 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([4.0, 4.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let resize6 = burn::nn::interpolate::Interpolate2dConfig::new()
            .with_output_size(None)
            .with_scale_factor(Some([2.0, 2.0]))
            .with_mode(burn::nn::interpolate::InterpolateMode::Nearest)
            .with_align_corners(false)
            .init();
        let conv2d122 = Conv2dConfig::new([256, 64], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(true)
            .init(device);
        let convtranspose2d1 = ConvTranspose2dConfig::new([64, 64], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant127: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 4>::zeros([1, 64, 1, 1], device),
            device.clone(),
            false,
            [1, 64, 1, 1].into(),
        );
        let convtranspose2d2 = ConvTranspose2dConfig::new([64, 1], [2, 2])
            .with_stride([2, 2])
            .with_padding([0, 0])
            .with_padding_out([0, 0])
            .with_dilation([1, 1])
            .with_groups(1)
            .with_bias(false)
            .init(device);
        let constant129: burn::module::Param<Tensor<B, 4>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
            move |device, _require_grad| Tensor::<B, 4>::zeros([1, 1, 1, 1], device),
            device.clone(),
            false,
            [1, 1, 1, 1].into(),
        );
        Self {
            conv2d55,
            conv2d56,
            conv2d57,
            conv2d58,
            resize1,
            resize2,
            resize3,
            conv2d59,
            conv2d60,
            conv2d61,
            conv2d62,
            conv2d63,
            conv2d64,
            conv2d65,
            conv2d66,
            conv2d67,
            conv2d68,
            conv2d69,
            conv2d70,
            conv2d71,
            conv2d72,
            conv2d73,
            conv2d74,
            conv2d75,
            conv2d76,
            conv2d77,
            conv2d78,
            conv2d79,
            conv2d80,
            conv2d81,
            conv2d82,
            conv2d83,
            conv2d84,
            conv2d85,
            conv2d86,
            conv2d87,
            conv2d88,
            conv2d89,
            conv2d90,
            conv2d91,
            conv2d92,
            conv2d93,
            conv2d94,
            conv2d95,
            conv2d96,
            conv2d97,
            conv2d98,
            conv2d99,
            conv2d100,
            conv2d101,
            conv2d102,
            conv2d103,
            conv2d104,
            conv2d105,
            conv2d106,
            conv2d107,
            conv2d108,
            conv2d109,
            conv2d110,
            conv2d111,
            conv2d112,
            conv2d113,
            conv2d114,
            conv2d115,
            conv2d116,
            conv2d117,
            conv2d118,
            conv2d119,
            conv2d120,
            conv2d121,
            resize4,
            resize5,
            resize6,
            conv2d122,
            convtranspose2d1,
            constant127,
            convtranspose2d2,
            constant129,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }
    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(
        &self,
        add23_out1: Tensor<B, 4>,
        add18_out1: Tensor<B, 4>,
        add9_out1: Tensor<B, 4>,
        add4_out1: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let conv2d55_out1 = self.conv2d55.forward(add23_out1);
        let conv2d56_out1 = self.conv2d56.forward(add18_out1);
        let conv2d57_out1 = self.conv2d57.forward(add9_out1);
        let conv2d58_out1 = self.conv2d58.forward(add4_out1);
        let resize1_out1 = self.resize1.forward(conv2d55_out1.clone());
        let add24_out1 = conv2d56_out1.add(resize1_out1);
        let resize2_out1 = self.resize2.forward(add24_out1.clone());
        let add25_out1 = conv2d57_out1.add(resize2_out1);
        let resize3_out1 = self.resize3.forward(add25_out1.clone());
        let add26_out1 = conv2d58_out1.add(resize3_out1);
        let conv2d59_out1 = self.conv2d59.forward(conv2d55_out1);
        let conv2d60_out1 = self.conv2d60.forward(conv2d59_out1);
        let conv2d61_out1 = self.conv2d61.forward(add24_out1);
        let conv2d62_out1 = self.conv2d62.forward(conv2d61_out1);
        let conv2d63_out1 = self.conv2d63.forward(add25_out1);
        let conv2d64_out1 = self.conv2d64.forward(conv2d63_out1);
        let conv2d65_out1 = self.conv2d65.forward(add26_out1);
        let conv2d66_out1 = self.conv2d66.forward(conv2d65_out1);
        let conv2d67_out1 = self.conv2d67.forward(conv2d66_out1.clone());
        let add27_out1 = conv2d64_out1.add(conv2d67_out1);
        let conv2d68_out1 = self.conv2d68.forward(add27_out1.clone());
        let add28_out1 = conv2d62_out1.add(conv2d68_out1);
        let conv2d69_out1 = self.conv2d69.forward(add28_out1.clone());
        let add29_out1 = conv2d60_out1.add(conv2d69_out1);
        let conv2d70_out1 = self.conv2d70.forward(conv2d66_out1);
        let conv2d71_out1 = self.conv2d71.forward(conv2d70_out1);
        let conv2d72_out1 = self.conv2d72.forward(add27_out1);
        let conv2d73_out1 = self.conv2d73.forward(conv2d72_out1);
        let conv2d74_out1 = self.conv2d74.forward(add28_out1);
        let conv2d75_out1 = self.conv2d75.forward(conv2d74_out1);
        let conv2d76_out1 = self.conv2d76.forward(add29_out1);
        let conv2d77_out1 = self.conv2d77.forward(conv2d76_out1);
        let conv2d78_out1 = self.conv2d78.forward(conv2d77_out1.clone());
        let conv2d79_out1 = self.conv2d79.forward(conv2d78_out1.clone());
        let conv2d80_out1 = self.conv2d80.forward(conv2d78_out1.clone());
        let conv2d81_out1 = self.conv2d81.forward(conv2d78_out1);
        let add30_out1 = conv2d79_out1.add(conv2d80_out1);
        let add31_out1 = add30_out1.add(conv2d81_out1);
        let conv2d82_out1 = self.conv2d82.forward(add31_out1.clone());
        let conv2d83_out1 = self.conv2d83.forward(add31_out1.clone());
        let conv2d84_out1 = self.conv2d84.forward(add31_out1);
        let add32_out1 = conv2d82_out1.add(conv2d83_out1);
        let add33_out1 = add32_out1.add(conv2d84_out1);
        let conv2d85_out1 = self.conv2d85.forward(add33_out1.clone());
        let conv2d86_out1 = self.conv2d86.forward(add33_out1.clone());
        let conv2d87_out1 = self.conv2d87.forward(add33_out1);
        let add34_out1 = conv2d85_out1.add(conv2d86_out1);
        let add35_out1 = add34_out1.add(conv2d87_out1);
        let conv2d88_out1 = self.conv2d88.forward(add35_out1);
        let relu11_out1 = burn::tensor::activation::relu(conv2d88_out1);
        let add36_out1 = conv2d77_out1.add(relu11_out1);
        let conv2d89_out1 = self.conv2d89.forward(conv2d75_out1.clone());
        let conv2d90_out1 = self.conv2d90.forward(conv2d89_out1.clone());
        let conv2d91_out1 = self.conv2d91.forward(conv2d89_out1.clone());
        let conv2d92_out1 = self.conv2d92.forward(conv2d89_out1);
        let add37_out1 = conv2d90_out1.add(conv2d91_out1);
        let add38_out1 = add37_out1.add(conv2d92_out1);
        let conv2d93_out1 = self.conv2d93.forward(add38_out1.clone());
        let conv2d94_out1 = self.conv2d94.forward(add38_out1.clone());
        let conv2d95_out1 = self.conv2d95.forward(add38_out1);
        let add39_out1 = conv2d93_out1.add(conv2d94_out1);
        let add40_out1 = add39_out1.add(conv2d95_out1);
        let conv2d96_out1 = self.conv2d96.forward(add40_out1.clone());
        let conv2d97_out1 = self.conv2d97.forward(add40_out1.clone());
        let conv2d98_out1 = self.conv2d98.forward(add40_out1);
        let add41_out1 = conv2d96_out1.add(conv2d97_out1);
        let add42_out1 = add41_out1.add(conv2d98_out1);
        let conv2d99_out1 = self.conv2d99.forward(add42_out1);
        let relu12_out1 = burn::tensor::activation::relu(conv2d99_out1);
        let add43_out1 = conv2d75_out1.add(relu12_out1);
        let conv2d100_out1 = self.conv2d100.forward(conv2d73_out1.clone());
        let conv2d101_out1 = self.conv2d101.forward(conv2d100_out1.clone());
        let conv2d102_out1 = self.conv2d102.forward(conv2d100_out1.clone());
        let conv2d103_out1 = self.conv2d103.forward(conv2d100_out1);
        let add44_out1 = conv2d101_out1.add(conv2d102_out1);
        let add45_out1 = add44_out1.add(conv2d103_out1);
        let conv2d104_out1 = self.conv2d104.forward(add45_out1.clone());
        let conv2d105_out1 = self.conv2d105.forward(add45_out1.clone());
        let conv2d106_out1 = self.conv2d106.forward(add45_out1);
        let add46_out1 = conv2d104_out1.add(conv2d105_out1);
        let add47_out1 = add46_out1.add(conv2d106_out1);
        let conv2d107_out1 = self.conv2d107.forward(add47_out1.clone());
        let conv2d108_out1 = self.conv2d108.forward(add47_out1.clone());
        let conv2d109_out1 = self.conv2d109.forward(add47_out1);
        let add48_out1 = conv2d107_out1.add(conv2d108_out1);
        let add49_out1 = add48_out1.add(conv2d109_out1);
        let conv2d110_out1 = self.conv2d110.forward(add49_out1);
        let relu13_out1 = burn::tensor::activation::relu(conv2d110_out1);
        let add50_out1 = conv2d73_out1.add(relu13_out1);
        let conv2d111_out1 = self.conv2d111.forward(conv2d71_out1.clone());
        let conv2d112_out1 = self.conv2d112.forward(conv2d111_out1.clone());
        let conv2d113_out1 = self.conv2d113.forward(conv2d111_out1.clone());
        let conv2d114_out1 = self.conv2d114.forward(conv2d111_out1);
        let add51_out1 = conv2d112_out1.add(conv2d113_out1);
        let add52_out1 = add51_out1.add(conv2d114_out1);
        let conv2d115_out1 = self.conv2d115.forward(add52_out1.clone());
        let conv2d116_out1 = self.conv2d116.forward(add52_out1.clone());
        let conv2d117_out1 = self.conv2d117.forward(add52_out1);
        let add53_out1 = conv2d115_out1.add(conv2d116_out1);
        let add54_out1 = add53_out1.add(conv2d117_out1);
        let conv2d118_out1 = self.conv2d118.forward(add54_out1.clone());
        let conv2d119_out1 = self.conv2d119.forward(add54_out1.clone());
        let conv2d120_out1 = self.conv2d120.forward(add54_out1);
        let add55_out1 = conv2d118_out1.add(conv2d119_out1);
        let add56_out1 = add55_out1.add(conv2d120_out1);
        let conv2d121_out1 = self.conv2d121.forward(add56_out1);
        let relu14_out1 = burn::tensor::activation::relu(conv2d121_out1);
        let add57_out1 = conv2d71_out1.add(relu14_out1);
        let resize4_out1 = self.resize4.forward(add36_out1);
        let resize5_out1 = self.resize5.forward(add43_out1);
        let resize6_out1 = self.resize6.forward(add50_out1);
        let concat2_out1 = burn::tensor::Tensor::cat(
            [resize4_out1, resize5_out1, resize6_out1, add57_out1].into(),
            1,
        );
        let conv2d122_out1 = self.conv2d122.forward(concat2_out1);
        let relu15_out1 = burn::tensor::activation::relu(conv2d122_out1);
        let convtranspose2d1_out1 = self.convtranspose2d1.forward(relu15_out1);
        let constant127_out1 = self.constant127.val();
        let add58_out1 = convtranspose2d1_out1.add(constant127_out1);
        let relu16_out1 = burn::tensor::activation::relu(add58_out1);
        let convtranspose2d2_out1 = self.convtranspose2d2.forward(relu16_out1);
        let constant129_out1 = self.constant129.val();
        let add59_out1 = convtranspose2d2_out1.add(constant129_out1);
        let sigmoid1_out1 = burn::tensor::activation::sigmoid(add59_out1);
        sigmoid1_out1
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    submodule1: Submodule1<B>,
    submodule2: Submodule2<B>,
    phantom: core::marker::PhantomData<B>,
    #[module(skip)]
    device: B::Device,
}


impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        Self::from_file(
            "/var/folders/r8/v0v7bpr93jn7636n86zxwn0r0000gn/T/nix-shell.kwhRLa/akuna-pp-ocr-codegen/burn/medium_det/medium_det.bpk",
            &Default::default(),
        )
    }
}

impl<B: Backend> Model<B> {
    /// Load model weights from a burnpack file.
    pub fn from_file(file: &str, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_file(file);
        model.load_from(&mut store).expect("Failed to load burnpack file");
        model
    }

    /// Load model weights from in-memory bytes.
    ///
    /// The bytes must be the contents of a `.bpk` file.
    pub fn from_bytes(bytes: Bytes, device: &B::Device) -> Self {
        let mut model = Self::new(device);
        let mut store = BurnpackStore::from_bytes(Some(bytes));
        model.load_from(&mut store).expect("Failed to load burnpack bytes");
        model
    }
}

impl<B: Backend> Model<B> {
    #[allow(unused_variables)]
    pub fn new(device: &B::Device) -> Self {
        let submodule1 = Submodule1::new(device);
        let submodule2 = Submodule2::new(device);
        Self {
            submodule1,
            submodule2,
            phantom: core::marker::PhantomData,
            device: device.clone(),
        }
    }

    #[allow(clippy::let_and_return, clippy::approx_constant)]
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (add23_out1, add18_out1, add9_out1, add4_out1) = self.submodule1.forward(x);
        let sigmoid1_out1 = self
            .submodule2
            .forward(add23_out1, add18_out1, add9_out1, add4_out1);
        sigmoid1_out1
    }
}
