#include <metal_stdlib>

using namespace metal;

template <typename T>
kernel void add_scalar(
    const device T *in,
    device T *out,
    constant T &scalar,
    constant uint &size,
    uint index [[thread_position_in_grid]]
) {
    if (index >= size) {
        return;
    }
    out[index] = in[index] + scalar;
}

template [[host_name("add_scalar__u8")]]
kernel void add_scalar<uchar>(device const uchar *, device uchar *, constant uchar &, constant uint &, uint);

template [[host_name("add_scalar__i32")]]
kernel void add_scalar<int>(device const int *, device int *, constant int &, constant uint &, uint);

template [[host_name("add_scalar__f16")]]
kernel void add_scalar<half>(device const half *, device half *, constant half &, constant uint &, uint);

template [[host_name("add_scalar__f32")]]
kernel void add_scalar<float>(device const float *, device float *, constant float &, constant uint &, uint);

template [[host_name("add_scalar__bf16")]]
kernel void add_scalar<bfloat>(device const bfloat *, device bfloat *, constant bfloat &, constant uint &, uint);
