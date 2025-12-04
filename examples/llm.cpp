#include "gguf.h"
#include "layer.h"
#include "mat.h"
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <sstream>

namespace ncnn {

class rope_module : public Layer
{
public:
    rope_module() { one_blob_only = true; }
    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
    {
        int seq_len = bottom_blob.h;
        int dim = bottom_blob.w;
        int half = dim / 2;
        top_blob.create(dim, seq_len);
        for (int i = 0; i < seq_len; i++) {
            const float* src = bottom_blob.row(i);
            float* dst = top_blob.row(i);
            for (int j = 0; j < half; j++) {
                float theta = powf(10000.0f, -2.0f * j / dim);
                float cos_t = cosf(i * theta);
                float sin_t = sinf(i * theta);
                dst[2*j] = src[2*j] * cos_t - src[2*j+1] * sin_t;
                dst[2*j+1] = src[2*j] * sin_t + src[2*j+1] * cos_t;
            }
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(rope_module)

} // namespace ncnn

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [gguf file] [token ids separated by space]\n", argv[0]);
        return -1;
    }

    ncnn::GGUFLoader loader;
    if (!loader.load(argv[1])) {
        fprintf(stderr, "Failed to load GGUF\n");
        return -1;
    }

    auto& kv_ints = loader.get_kv_ints();
    int n_layers = kv_ints.at("phi3.block_count");
    int n_head = kv_ints.at("phi3.attention.head_count");
    int n_kv_head = kv_ints.at("phi3.attention.head_count_kv");
    int hidden_size = kv_ints.at("phi3.embedding_length");
    int vocab_size = kv_ints.at("phi3.vocab_size");

    std::unordered_map<std::string, ncnn::Mat> weights;
    for (auto& p : loader.get_tensor_map()) {
        weights[p.first] = dequant_gguf_tensor(p.second, loader.get_file_data());
    }

    std::string prompt = argv[2];
    std::vector<int> tokens;
    std::stringstream ss(prompt);
    int id;
    while (ss >> id) tokens.push_back(id);

    ncnn::Option opt;
    opt.use_vulkan_compute = true;

    ncnn::Mat x(hidden_size, tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        memcpy(x.row(i), weights["phi3.embed_tokens"].row(tokens[i]), hidden_size * sizeof(float));
    }

    for (int l = 0; l < n_layers; l++) {
        ncnn::LayerNorm norm;
        norm.affine = true;
        norm.eps = 1e-5f;
        norm.weight_data = weights["phi3.layers." + std::to_string(l) + ".input_layernorm.weight"];
        norm.bias_data = weights["phi3.layers." + std::to_string(l) + ".input_layernorm.bias"];
        ncnn::Mat norm_out;
        norm.forward(x, norm_out, opt);

        ncnn::InnerProduct ip_q;
        ip_q.weight_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.q_proj.weight"];
        ip_q.bias_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.q_proj.bias"];
        ncnn::Mat q;
        ip_q.forward(norm_out, q, opt);

        ncnn::InnerProduct ip_k;
        ip_k.weight_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.k_proj.weight"];
        ip_k.bias_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.k_proj.bias"];
        ncnn::Mat k;
        ip_k.forward(norm_out, k, opt);

        ncnn::InnerProduct ip_v;
        ip_v.weight_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.v_proj.weight"];
        ip_v.bias_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.v_proj.bias"];
        ncnn::Mat v;
        ip_v.forward(norm_out, v, opt);

        int head_dim = hidden_size / n_head;
        ncnn::Mat attn_out(hidden_size, tokens.size());
        for (int h = 0; h < n_head; h++) {
            int offset = h * head_dim;
            ncnn::Mat q_h(head_dim, tokens.size());
            for (int s = 0; s < tokens.size(); s++) {
                memcpy(q_h.row(s), q.row(s) + offset, head_dim * sizeof(float));
            }
            ncnn::Mat k_h(head_dim, tokens.size());
            for (int s = 0; s < tokens.size(); s++) {
                memcpy(k_h.row(s), k.row(s) + offset, head_dim * sizeof(float));
            }
            ncnn::Mat v_h(head_dim, tokens.size());
            for (int s = 0; s < tokens.size(); s++) {
                memcpy(v_h.row(s), v.row(s) + offset, head_dim * sizeof(float));
            }

            ncnn::rope_module rope;
            ncnn::Mat q_rot;
            rope.forward(q_h, q_rot, opt);
            ncnn::Mat k_rot;
            rope.forward(k_h, k_rot, opt);

            ncnn::Mat scores(tokens.size(), tokens.size());
            for (int i = 0; i < tokens.size(); i++) {
                for (int j = 0; j < tokens.size(); j++) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q_rot.row(i)[d] * k_rot.row(j)[d];
                    }
                    scores.row(i)[j] = dot / sqrtf(head_dim);
                }
            }

            for (int i = 0; i < tokens.size(); i++) {
                float maxv = *std::max_element(scores.row(i), scores.row(i) + tokens.size());
                float sum = 0;
                for (int j = 0; j < tokens.size(); j++) {
                    scores.row(i)[j] = expf(scores.row(i)[j] - maxv);
                    sum += scores.row(i)[j];
                }
                for (int j = 0; j < tokens.size(); j++) {
                    scores.row(i)[j] /= sum;
                }
            }

            ncnn::Mat out_h(head_dim, tokens.size());
            for (int i = 0; i < tokens.size(); i++) {
                for (int d = 0; d < head_dim; d++) {
                    float val = 0;
                    for (int j = 0; j < tokens.size(); j++) {
                        val += scores.row(i)[j] * v_h.row(j)[d];
                    }
                    out_h.row(i)[d] = val;
                }
            }

            for (int s = 0; s < tokens.size(); s++) {
                memcpy(attn_out.row(s) + offset, out_h.row(s), head_dim * sizeof(float));
            }
        }

        ncnn::InnerProduct ip_o;
        ip_o.weight_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.o_proj.weight"];
        ip_o.bias_data = weights["phi3.layers." + std::to_string(l) + ".self_attn.o_proj.bias"];
        ncnn::Mat attn_proj;
        ip_o.forward(attn_out, attn_proj, opt);

        for (int i = 0; i < x.total(); i++) {
            x[i] += attn_proj[i];
        }

        ncnn::LayerNorm post_norm;
        post_norm.affine = true;
        post_norm.eps = 1e-5f;
        post_norm.weight_data = weights["phi3.layers." + std::to_string(l) + ".post_attention_layernorm.weight"];
        post_norm.bias_data = weights["phi3.layers." + std::to_string(l) + ".post_attention_layernorm.bias"];
        ncnn::Mat post_norm_out;
        post_norm.forward(x, post_norm_out, opt);

        ncnn::InnerProduct ip_gate;
        ip_gate.weight_data = weights["phi3.layers." + std::to_string(l) + ".mlp.gate_proj.weight"];
        ip_gate.bias_data = weights["phi3.layers." + std::to_string(l) + ".mlp.gate_proj.bias"];
        ncnn::Mat gate;
        ip_gate.forward(post_norm_out, gate, opt);

        ncnn::InnerProduct ip_up;
        ip_up.weight_data = weights["phi3.layers." + std::to_string(l) + ".mlp.up_proj.weight"];
        ip_up.bias_data = weights["phi3.layers." + std::to_string(l) + ".mlp.up_proj.bias"];
        ncnn::Mat up;
        ip_up.forward(post_norm_out, up, opt);

        ncnn::Mat mlp_hidden(gate.w, gate.h);
        for (int i = 0; i < gate.total(); i++) {
            mlp_hidden[i] = gate[i] * (1 / (1 + expf(-gate[i]))) * up[i];
        }

        ncnn::InnerProduct ip_down;
        ip_down.weight_data = weights["phi3.layers." + std::to_string(l) + ".mlp.down_proj.weight"];
        ip_down.bias_data = weights["phi3.layers." + std::to_string(l) + ".mlp.down_proj.bias"];
        ncnn::Mat mlp_out;
        ip_down.forward(mlp_hidden, mlp_out, opt);

        for (int i = 0; i < x.total(); i++) {
            x[i] += mlp_out[i];
        }
    }

    ncnn::LayerNorm final_norm;
    final_norm.affine = true;
    final_norm.eps = 1e-5f;
    final_norm.weight_data = weights["phi3.norm.weight"];
    final_norm.bias_data = weights["phi3.norm.bias"];
    ncnn::Mat norm_x;
    final_norm.forward(x, norm_x, opt);

    ncnn::InnerProduct lm_head;
    lm_head.weight_data = weights["lm_head.weight"];
    lm_head.bias_data = weights["lm_head.bias"];
    ncnn::Mat logits(vocab_size, tokens.size());
    lm_head.forward(norm_x, logits, opt);

    float* last_logits = logits.row(tokens.size() - 1);
    int max_id = 0;
    float max_val = last_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (last_logits[i] > max_val) {
            max_val = last_logits[i];
            max_id = i;
        }
    }

    std::cout << "Predicted token id: " << max_id << std::endl;

    return 0;
}