# Feature Request-based Refactoring Recommendation

## Introduction
In software development, adapting to changing requirements is crucial for the advancement of software systems. Feature requests play a significant role in enhancing software by adding new functionalities or improving existing ones. However, determining the appropriate type of refactoring to accommodate these requests can be challenging. This project aims to address this challenge by proposing a learning-based technique to suggest refactoring types based on the history of feature requests, implemented refactorings, and code smells.

## Contributions
1. Reproduced and critically evaluated the original work by Nyamawe et al. (2020) and identified potential flaws in the implementation.
2. Proposed a new method, d-BERT, incorporating sequential modeling to better understand the context of feature request documentation and improve performance in predicting the need for refactoring.
3. Introduced newer, model-agnostic metrics like SHAP to evaluate the performance of different architectures in a fair and unbiased manner.

## Proposed Solution
### Reproducibility
- Re-implemented all experiments from the original paper to understand parameter selection and hyperparameter tuning.
- Examined the impact of parameters on both PMD applications and machine learning models.

### Improvements
- Proposed using attention-based methods like BERT to encode feature request documentation, preserving contextual information.
- Explored ensemble classifiers such as AdaBoost and Gradient Boosted Trees to enhance the classification performance.

### Extensibility
- Extended the scope of the study to apply the proposed approach to new codebases, providing insights into its generalizability and performance across different projects.

## Conclusion
- The original work provided a novel approach to recommending refactoring types based on feature requests, code smells, and refactorings.
- We proposed improvements and extensions to enhance the performance and applicability of the approach.
- Our preliminary findings suggest that newer models like d-BERT could outperform traditional machine learning classifiers in predicting refactoring needs.

## Future Work
- Further explore the use of large language models like GPT-3, PaLM, and BLOOM for predicting refactorings based on feature request documentation.
- Investigate the feasibility of automating the refactoring process using these models to suggest code changes aligned with feature requests.

## Tools Used
- Python
- TensorFlow
- PyTorch
- Scikit-learn
- SHAP
- BERT
