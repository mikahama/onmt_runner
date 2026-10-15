// Compatibility implementation of TensorFlow Addons' deprecated GatherTree
// op. OpenNMT-tf 2.0 SavedModels commonly reference it as Addons>GatherTree.
#include <algorithm>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("Addons>GatherTree")
    .Input("step_ids: T")
    .Input("parent_ids: T")
    .Input("max_sequence_lengths: T")
    .Input("end_token: T")
    .Output("beams: T")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle step_ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &step_ids));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &step_ids));
      c->set_output(0, c->input(0));
      return absl::OkStatus();
    });

template <typename T>
class GatherTreeOp : public OpKernel {
 public:
  explicit GatherTreeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& step_ids_tensor = context->input(0);
    const Tensor& parent_ids_tensor = context->input(1);
    const Tensor& lengths_tensor = context->input(2);
    const Tensor& end_tensor = context->input(3);
    OP_REQUIRES(context, step_ids_tensor.dims() == 3,
                errors::InvalidArgument("step_ids must have rank 3"));
    OP_REQUIRES(context, parent_ids_tensor.shape() == step_ids_tensor.shape(),
                errors::InvalidArgument("parent_ids must match step_ids"));
    const int max_time = step_ids_tensor.dim_size(0);
    const int batch_size = step_ids_tensor.dim_size(1);
    const int beam_width = step_ids_tensor.dim_size(2);
    OP_REQUIRES(context, lengths_tensor.dims() == 1 &&
                             lengths_tensor.dim_size(0) == batch_size,
                errors::InvalidArgument("max_sequence_lengths must have shape [batch]"));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(end_tensor.shape()),
                errors::InvalidArgument("end_token must be a scalar"));
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, step_ids_tensor.shape(), &output_tensor));
    auto step_ids = step_ids_tensor.tensor<T, 3>();
    auto parent_ids = parent_ids_tensor.tensor<T, 3>();
    auto lengths = lengths_tensor.vec<T>();
    auto output = output_tensor->tensor<T, 3>();
    const T end_token = end_tensor.scalar<T>()();

    for (int batch = 0; batch < batch_size; ++batch) {
      for (int beam = 0; beam < beam_width; ++beam) {
        const int length = std::max(0, std::min(max_time, static_cast<int>(lengths(batch))));
        int parent = beam;
        for (int time = length - 1; time >= 0; --time) {
          OP_REQUIRES(context, parent >= 0 && parent < beam_width,
                      errors::InvalidArgument("parent index outside beam width"));
          output(time, batch, beam) = step_ids(time, batch, parent);
          parent = static_cast<int>(parent_ids(time, batch, parent));
        }
        for (int time = length; time < max_time; ++time)
          output(time, batch, beam) = end_token;
        bool finished = false;
        for (int time = 0; time < max_time; ++time) {
          if (finished)
            output(time, batch, beam) = end_token;
          else if (output(time, batch, beam) == end_token)
            finished = true;
        }
      }
    }
  }
};

#define REGISTER_KERNEL(type) \
  REGISTER_KERNEL_BUILDER(Name("Addons>GatherTree").Device(DEVICE_CPU).TypeConstraint<type>("T"), GatherTreeOp<type>)
REGISTER_KERNEL(int32);
REGISTER_KERNEL(int64_t);
#undef REGISTER_KERNEL
