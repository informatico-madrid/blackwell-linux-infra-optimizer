# 📊 Performance Analysis: DeepSeek-R1 32B on Blackwell

## 🎯 Test Scenario: Sovereign Legal RAG Design
- **Engine**: vLLM (FlashInfer Backend)
- **Model**: DeepSeek-R1-Distill-Qwen-32B-AWQ
- **Hardware**: Dual RTX 5090 (64GB Total VRAM) via PCIe Gen 5

## 📈 Real-Time Metrics
| Metric | Value |
| :--- | :--- |
| **Generation Throughput** | **59.0 tokens/s** |
| **Prefix Cache Hit Rate** | **44.4%** |
| **VRAM Consumption** | 30.1 GB / GPU (Optimized KV Cache) |

## 🧠 Reasoning Execution
The model successfully executed a high-complexity architectural design. 
- **Total Tokens**: 1485
- **Reasoning (<think>)**: ~116 tokens for strategy validation.
- **Finish Reason**: `stop` (Full execution without context truncation).

> [Link to raw JSON response](./logs/r1-legal-rag-response.json)
