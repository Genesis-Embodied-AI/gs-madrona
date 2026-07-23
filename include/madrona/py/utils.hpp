#pragma once
#include <madrona/macros.hpp>
#include <madrona/span.hpp>
#include <madrona/optional.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/dyn_array.hpp>

#include <array>

// All the below classes are virtual because nanobind
// uses RTTI to match types across modules which only works with virtual types

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/cuda_utils.hpp>
#endif

namespace madrona::py {

enum class TensorElementType {
    UInt8,
    Int8,
    Int16,
    Int32,
    Int64,
    Float16,
    Float32,
};

struct TensorInterface {
    TensorElementType type;
    Span<const int64_t> dimensions;
};

class Tensor final {
public:
    static inline constexpr int64_t maxDimensions = 16;

    class Printer {
    public:
        Printer(const Printer &) = delete;
        Printer(Printer &&o);
        ~Printer();

        void print(int64_t flatten_dim = 0) const;

    private:
        int64_t printInnerDims(void *print_ptr,
                               int64_t num_inner_items,
                               int64_t cur_offset) const;

        int64_t printOuterDim(int64_t dim,
                              int64_t flatten_dim,
                              void *print_ptr,
                              int64_t num_inner_items,
                              int64_t cur_offset) const;

        inline Printer(void *dev_ptr,
                       void *print_ptr,
                       TensorElementType type,
                       Span<const int64_t> dimensions,
                       int64_t num_total_bytes);

        void *dev_ptr_;
        void *print_ptr_;
        TensorElementType type_;
        int64_t num_dimensions_;
        std::array<int64_t, maxDimensions> dimensions_;
        int64_t num_total_bytes_;

    friend class Tensor;
    };

    Tensor(void *dev_ptr, TensorElementType type,
           Span<const int64_t> dimensions,
           Optional<int> gpu_id);

    Tensor(const Tensor &o);
    Tensor & operator=(const Tensor &o);
    static Tensor none();
    inline bool isNone() const { return is_none_; }
    inline void * devicePtr() const { return dev_ptr_; }
    inline TensorElementType type() const { return type_; }
    inline bool isOnGPU() const { return gpu_id_ != -1; }
    inline int gpuID() const { return gpu_id_; }
    inline int64_t numDims() const { return num_dimensions_; }
    inline const int64_t *dims() const { return dimensions_.data(); }
    int64_t numBytesPerItem() const;

    TensorInterface interface() const;

    Printer makePrinter() const;
private:
#ifdef MADRONA_LINUX
    virtual void key_();
#endif

    void *dev_ptr_;
    TensorElementType type_;
    int gpu_id_;
    bool is_none_ = false;

    int64_t num_dimensions_;
    std::array<int64_t, maxDimensions> dimensions_;
};

}
