#include <madrona/py/utils.hpp>
#include <madrona/heap_array.hpp>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

#include <cassert>
#include <cstring>
#include <cstdio>
#include <string>

namespace madrona::py {

Tensor::Printer::Printer(Printer &&o)
    : dev_ptr_(o.dev_ptr_),
      print_ptr_(o.print_ptr_)
{
    o.print_ptr_ = nullptr;
}

Tensor::Printer::~Printer()
{
    if (print_ptr_ == nullptr) {
        return;
    }

#ifdef MADRONA_CUDA_SUPPORT
    cu::deallocCPU(print_ptr_);
#endif
}

void Tensor::Printer::print(int64_t flatten_dim) const
{
    void *print_ptr;
    if (print_ptr_ == nullptr) {
        print_ptr = dev_ptr_;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(print_ptr_, dev_ptr_,
                   num_total_bytes_,
                   cudaMemcpyDeviceToHost);
#else
        (void)num_total_bytes_;
        FATAL("Trying to print CUDA tensor, no CUDA support");
#endif
        print_ptr = print_ptr_;
    }

    int64_t num_inner_items = 1;
    for (int64_t i = flatten_dim + 1; i < num_dimensions_; i++) {
        int64_t dim_size = dimensions_[i];
        num_inner_items *= dim_size;
    }

    printOuterDim(0, flatten_dim, print_ptr, num_inner_items, 0);
}

int64_t Tensor::Printer::printInnerDims(void *print_ptr,
                                        int64_t num_inner_items,
                                        int64_t cur_offset) const
{
    switch (type_) {
    case TensorElementType::Int32: {
        auto base = (int32_t *)print_ptr + cur_offset;
    
        for (int64_t i = 0; i < num_inner_items; i++) {
            printf("%d ", base[i]);
        }
    } break;
    case TensorElementType::Float32: {
        auto base = (float *)print_ptr + cur_offset;
    
        for (int64_t i = 0; i < num_inner_items; i++) {
            printf("%.3f ", base[i]);
        }
    } break;
    default: break;
    }

    return cur_offset + num_inner_items;
}

int64_t Tensor::Printer::printOuterDim(int64_t dim,
                                       int64_t flatten_dim,
                                       void *print_ptr,
                                       int64_t num_inner_items,
                                       int64_t cur_offset) const
{
    int64_t dim_size = dimensions_[dim];
    if (dim == flatten_dim) {
        for (CountT i = 0; i < dim_size; i++) {
            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }

            if (num_dimensions_ - flatten_dim > 1) {
                printf("[ ");
            }
            cur_offset = printInnerDims(
                print_ptr, num_inner_items, cur_offset);

            if (num_dimensions_ - flatten_dim > 1) {
                printf("]");
            }
            printf("\n");
        }
    } else {
        for (CountT i = 0; i < dim_size; i++) {
            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }

            printf("[\n");
            cur_offset = printOuterDim(dim + 1, flatten_dim, print_ptr,
                                       num_inner_items, cur_offset);

            for (int64_t j = 0; j < dim; j++) {
                printf("  ");
            }
            printf("]\n");
        }
    }

    return cur_offset;
}

Tensor::Printer::Printer(void *dev_ptr,
                         void *print_ptr,
                         TensorElementType type,
                         Span<const int64_t> dimensions,
                         int64_t num_total_bytes)
    : dev_ptr_(dev_ptr),
      print_ptr_(print_ptr),
      type_(type),
      num_dimensions_(dimensions.size()),
      dimensions_(),
      num_total_bytes_(num_total_bytes)
{
    for (int64_t i = 0; i < num_dimensions_; i++) {
        dimensions_[i] = dimensions[i];
    }
}

Tensor::Tensor(void *dev_ptr, TensorElementType type,
                              Span<const int64_t> dimensions,
                              Optional<int> gpu_id)
    : dev_ptr_(dev_ptr),
      type_(type),
      gpu_id_(gpu_id.has_value() ? *gpu_id : -1),
      num_dimensions_(dimensions.size()),
      dimensions_()
{
    assert(num_dimensions_ <= maxDimensions);
    memcpy(dimensions_.data(), dimensions.data(),
           num_dimensions_ * sizeof(int64_t));
}

Tensor::Tensor(const Tensor &o)
    : dev_ptr_(o.dev_ptr_),
      type_(o.type_),
      gpu_id_(o.gpu_id_),
      is_none_(o.is_none_),
      num_dimensions_(o.num_dimensions_),
      dimensions_(o.dimensions_)
{}

Tensor Tensor::none()
{
    auto res = Tensor(nullptr, TensorElementType::Float32, {}, -1);
    res.is_none_ = true;
    return res;
}

Tensor & Tensor::operator=(const Tensor &o)
{
    dev_ptr_ = o.dev_ptr_;
    type_ = o.type_;
    gpu_id_ = o.gpu_id_;
    num_dimensions_ = o.num_dimensions_;
    dimensions_ = o.dimensions_;

    return *this;
}

Tensor::Printer Tensor::makePrinter() const
{
    int64_t num_total_items = dimensions_[num_dimensions_ - 1];
    for (int64_t i = num_dimensions_ - 2; i >= 0; i--) {
        num_total_items *= dimensions_[i];
    }
    int64_t num_total_bytes = num_total_items * numBytesPerItem();

    void *print_ptr;
    if (!isOnGPU()) {
        print_ptr = nullptr;
    } else {
#ifdef MADRONA_CUDA_SUPPORT
        print_ptr = cu::allocReadback(num_total_bytes);
#else
        print_ptr = nullptr;
#endif
    }

    return Printer {
        dev_ptr_,
        print_ptr,
        type_,
        Span(dimensions_.data(), num_dimensions_),
        num_total_bytes,
    };
}

int64_t Tensor::numBytesPerItem() const
{
    switch (type_) {
        case TensorElementType::UInt8: return 1;
        case TensorElementType::Int8: return 1;
        case TensorElementType::Int16: return 2;
        case TensorElementType::Int32: return 4;
        case TensorElementType::Int64: return 8;
        case TensorElementType::Float16: return 2;
        case TensorElementType::Float32: return 4;
        default: return 0;
    }
}

TensorInterface Tensor::interface() const
{
    return TensorInterface {
        .type = type_,
        .dimensions = Span<const int64_t>(dimensions_.data(), num_dimensions_),
    };
}

#ifdef MADRONA_LINUX
void Tensor::key_() {}
#endif

}
