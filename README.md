# FEU-NET
This is the official code for FEU-NET: A Fine-grained Entity Understanding Network for Weakly Supervised Phrase Grounding


## 📄 Abstract

Weakly supervised phrase grounding (WSPG) aims to localize visual objects referred to by textual phrases without relying on region-level bounding box annotations. However, existing methods often struggle with linguistic diversity and fail to determine whether the described object is present in the image. To address these challenges, we propose FEU-NET, a fine-grained entity understanding network for WSPG. Firstly, a coarse-grained alignment (CGA) module aligns visual features with the global context of the phrase to identify potential object candidates. Secondly, a fine-grained alignment (FGA) strategy leverages large language models (LLMs) to generate counterfactual and paraphrased phrases. It enhances the model’s robustness to linguistic variations through contrastive learning. A prediction head then determines the presence of the target object, and grounding heatmaps are generated as intermediate representations for the final localization. In addition, we utilize visual prompt engineering (VPE) by using the CGA module output as a spatial prior, effectively constraining the search space for fine-grained entity grounding. Extensive experiments on five benchmark datasets demonstrate that our method achieves state-of-the-art performance.

<div align="center"> <img src="https://github.com/user-attachments/assets/2e90470e-e97e-42d6-877f-9d12c129fbc4" alt="FEU-NET Framework Overview" width="500" /> </div>

## 🚀 Quick Start (Coming Soon)


## 🧪 Benchmark Dataset (Coming Soon)


## 🛠️ License
This project is released under the MIT License.
