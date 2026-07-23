#include <madrona/py/bindings.hpp>
#include <madrona/crash.hpp>

namespace nb = nanobind;

namespace madrona::py {

namespace {

nb::dlpack::dtype toDLPackType(TensorElementType type)
{
    switch (type) {
        case TensorElementType::UInt8:
            return nb::dtype<uint8_t>();
        case TensorElementType::Int8:
            return nb::dtype<int8_t>();
        case TensorElementType::Int16:
            return nb::dtype<int16_t>();
        case TensorElementType::Int32:
            return nb::dtype<int32_t>();
        case TensorElementType::Int64:
            return nb::dtype<int64_t>();
        case TensorElementType::Float16:
            return nb::dlpack::dtype {
                static_cast<uint8_t>(nb::dlpack::dtype_code::Float),
                sizeof(int16_t) * 8,
                1,
            };
        case TensorElementType::Float32:
            return nb::dtype<float>();
        default: MADRONA_UNREACHABLE();
    }
}

nb::object tensor_to_pytorch(const Tensor &tensor)
{
    if (tensor.isNone()) {
        return nb::none();
    }

    nb::dlpack::dtype type = toDLPackType(tensor.type());

    return nb::cast(nb::ndarray<nb::pytorch>(
        tensor.devicePtr(),
        (size_t)tensor.numDims(),
        (const size_t *)tensor.dims(),
        nb::handle(),
        nullptr,
        type,
        tensor.isOnGPU() ?
            nb::device::cuda::value :
            nb::device::cpu::value,
        tensor.isOnGPU() ? tensor.gpuID() : 0
    ));
}

}

void setupMadronaSubmodule(nb::module_ parent_mod)
{
    auto m = parent_mod.def_submodule("madrona");

    nb::class_<Tensor>(m, "Tensor")
        .def("to_torch", tensor_to_pytorch, nb::rv_policy::automatic_reference)
    ;
}

}
