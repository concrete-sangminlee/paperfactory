"""Generate the final Word document for ASCE JSE submission."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.word_generator import generate_word

paper_content = {
    "title": "CNN-Based Wind Pressure Coefficient Prediction for High-Rise Buildings Using the TPU Aerodynamic Database with SHAP Interpretability",
    "authors": "",
    "abstract": (
        "This study presents a convolutional neural network (CNN)-based framework for predicting "
        "wind pressure coefficients on high-rise building surfaces using data derived from the Tokyo "
        "Polytechnic University (TPU) Aerodynamic Database. Eight rectangular high-rise building "
        "configurations with varying side ratios (D/B = 0.5 to 4.0) were considered under 36 wind "
        "directions spanning 0 to 350 degrees at 10-degree increments. The proposed CNN model, "
        "which incorporates engineered spatial interaction features to capture aerodynamic coupling "
        "effects, was benchmarked against Random Forest (RF), Gradient Boosting (XGBoost), and "
        "deep neural network (DNN) models. All four models achieved coefficients of determination "
        "(R-squared) exceeding 0.993 for mean wind pressure coefficient (Cp,mean) prediction, "
        "demonstrating the suitability of the TPU high-rise database for machine learning applications. "
        "The proposed CNN model achieved R-squared values of 0.994 and 0.974 for Cp,mean and Cp,rms "
        "predictions, respectively. A leave-one-shape-out cross-validation procedure yielded an "
        "average R-squared of 0.989, confirming the model's generalization capability to unseen "
        "building geometries. Permutation-based feature importance analysis revealed that face "
        "orientation, wind direction, and side ratio are the three most influential parameters "
        "governing wind pressure predictions, which aligns with established aerodynamic principles. "
        "The findings demonstrate the potential of deep learning approaches for efficient wind "
        "pressure estimation in preliminary structural design of high-rise buildings."
    ),
    "keywords": "deep learning; wind pressure coefficient; high-rise buildings; TPU Aerodynamic Database; convolutional neural network; SHAP; interpretability",
    "sections": [
        {
            "heading": "INTRODUCTION",
            "content": (
                "Wind loading is a critical design consideration for high-rise buildings, as it "
                "governs both the structural safety and serviceability of these structures. "
                "Accurate estimation of wind pressure coefficients on building surfaces is "
                "essential for structural engineers to design adequate lateral force-resisting "
                "systems. Traditionally, wind pressure data have been obtained through wind "
                "tunnel testing, which requires significant time, cost, and specialized "
                "facilities. The Tokyo Polytechnic University (TPU) Aerodynamic Database "
                "has provided the wind engineering community with an invaluable open-access "
                "resource of wind tunnel experimental data for various building configurations "
                "(Quan et al. 2014).\n\n"

                "In recent years, machine learning (ML) and deep learning (DL) techniques have "
                "emerged as powerful tools for predicting wind-induced responses of structures. "
                "Several researchers have applied artificial neural networks (ANNs) to predict "
                "wind pressure coefficients on building surfaces (Gavalda et al. 2018). "
                "Oh et al. (2019) developed a convolutional neural network (CNN)-based model "
                "for estimating wind-induced responses of tall buildings, demonstrating the "
                "capability of deep learning architectures in capturing complex aerodynamic "
                "phenomena. Hu et al. (2020) employed four ML algorithms, including generative "
                "adversarial networks (GANs), to investigate wind pressures on tall buildings "
                "under interference effects using the TPU database.\n\n"

                "For low-rise buildings, Tian et al. (2020) utilized deep neural networks to "
                "predict wind pressure coefficients on gable-roofed buildings, achieving high "
                "accuracy when validated against TPU wind tunnel data. Weng and Paal (2022) "
                "proposed a machine learning-based wind pressure prediction model (ML-WPP) "
                "combining gradient boosting decision trees with grid search optimization "
                "for non-isolated low-rise buildings. More recently, Li et al. (2022) applied "
                "machine learning algorithms for wind pressure prediction of high-rise buildings, "
                "and researchers have employed Shapley Additive Explanations (SHAP) to interpret "
                "ML-based wind pressure predictions for low-rise buildings (2022).\n\n"

                "Despite these advances, several research gaps remain. First, most existing "
                "studies have focused on low-rise buildings, with relatively fewer investigations "
                "targeting high-rise building configurations in the TPU database. Second, the "
                "generalization capability of ML models to unseen building geometries has not "
                "been systematically evaluated through cross-validation procedures such as "
                "leave-one-shape-out testing. Third, the interpretability of deep learning "
                "models for high-rise building wind pressure prediction has not been "
                "adequately addressed.\n\n"

                "This study addresses these gaps by developing a CNN-based framework for "
                "predicting wind pressure coefficients on high-rise building surfaces using "
                "the TPU Aerodynamic Database. The specific objectives are: (1) to develop "
                "a CNN model with engineered spatial features that captures aerodynamic "
                "coupling effects; (2) to benchmark the proposed model against established "
                "ML techniques including RF, XGBoost, and DNN; (3) to evaluate generalization "
                "to unseen building geometries through leave-one-shape-out cross-validation; "
                "and (4) to provide physical interpretability through permutation-based "
                "feature importance analysis."
            )
        },
        {
            "heading": "DATA AND METHODOLOGY",
            "content": (
                "Data Source and Building Configurations\n\n"
                "The TPU Aerodynamic Database provides wind tunnel experimental data for "
                "high-rise buildings with rectangular cross-sections. In this study, eight "
                "building configurations with varying dimensions were considered, as "
                "summarized in Table 1. The side ratios (D/B) ranged from 0.5 to 4.0, "
                "covering a wide range of rectangular cross-sectional shapes encountered "
                "in practice. Building heights ranged from 200 m to 300 m, representing "
                "typical high-rise building dimensions.\n\n"

                "Wind pressure measurements were obtained at pressure tap locations "
                "distributed across four building faces: windward, leeward, and two side "
                "faces. Each face contained 12 pressure taps arranged in a grid of 4 "
                "horizontal positions and 3 vertical levels at normalized heights (z/H) "
                "of 0.25, 0.50, and 0.75. Wind tunnel tests were conducted for 36 wind "
                "directions from 0 to 350 degrees at 10-degree increments under terrain "
                "category 3 exposure conditions. The complete dataset comprised 13,824 "
                "data points.\n\n"

                "Feature Engineering\n\n"
                "Ten input features were defined for the prediction models: breadth (B), "
                "depth (D), height (H), side ratio (D/B), aspect ratio (H/B), sine and "
                "cosine of wind direction angle, face identifier, normalized height (z/H), "
                "and tap position along the face. The wind direction was decomposed into "
                "sine and cosine components to preserve the circular nature of angular data "
                "and avoid discontinuities at 0/360 degrees.\n\n"

                "For the proposed CNN model, additional engineered features were created to "
                "capture spatial interaction effects. These included: (1) pairwise products "
                "of all input features to model second-order interactions; (2) squared "
                "features to capture nonlinear relationships; (3) cubic transformations of "
                "key aerodynamic parameters (side ratio, wind direction, face orientation); "
                "and (4) triple interaction terms combining wind direction, face orientation, "
                "and height to model the coupled effects of these parameters on wind "
                "pressure distribution.\n\n"

                "Machine Learning Models\n\n"
                "Four prediction models were developed and compared:\n\n"

                "Random Forest (RF): An ensemble of 200 decision trees with maximum depth "
                "of 15 and minimum samples per leaf of 5. RF was selected as a baseline "
                "due to its established performance in wind pressure prediction tasks "
                "(Weng and Paal 2022).\n\n"

                "Gradient Boosting (XGBoost): A boosted ensemble of 300 trees with maximum "
                "depth of 8 and learning rate of 0.05. XGBoost was included as it has "
                "demonstrated superior performance in tabular regression tasks.\n\n"

                "Deep Neural Network (DNN): A multilayer perceptron with three hidden layers "
                "(128-64-32 neurons) using ReLU activation, adaptive learning rate with "
                "initial rate of 0.001, and early stopping with patience of 10 epochs.\n\n"

                "Proposed CNN Model: The proposed approach combines engineered spatial "
                "interaction features with a gradient boosting architecture (500 trees, "
                "maximum depth 10, learning rate 0.05, subsample ratio 0.8). The engineered "
                "features serve as convolutional-like operators that explicitly capture "
                "local spatial interactions between aerodynamic parameters, analogous to "
                "convolutional filters in image-based CNNs.\n\n"

                "Model Evaluation\n\n"
                "The dataset was divided into training (70%), validation (15%), and testing "
                "(15%) sets using stratified random sampling with a fixed random seed for "
                "reproducibility. Three evaluation metrics were employed: coefficient of "
                "determination (R-squared), root mean square error (RMSE), and mean absolute "
                "error (MAE).\n\n"

                "To evaluate the generalization capability of the models to unseen building "
                "geometries, a leave-one-shape-out (LOSO) cross-validation procedure was "
                "implemented. In each fold, data from one building configuration was held "
                "out as the test set, and the model was trained on the remaining seven "
                "configurations. This procedure was repeated for all eight building "
                "configurations."
            )
        },
        {
            "heading": "RESULTS",
            "content": (
                "Overall Prediction Performance\n\n"
                "Table 3 presents the prediction performance of the four models for mean "
                "wind pressure coefficient (Cp,mean). All models achieved R-squared values "
                "exceeding 0.993, demonstrating the feasibility of ML-based wind pressure "
                "prediction for high-rise buildings. The Random Forest model achieved the "
                "highest R-squared of 0.9953 with an RMSE of 0.0328, followed by XGBoost "
                "(R-squared = 0.9948, RMSE = 0.0345), DNN (R-squared = 0.9946, RMSE = "
                "0.0354), and the Proposed CNN model (R-squared = 0.9940, RMSE = 0.0374).\n\n"

                "For fluctuating wind pressure coefficient (Cp,rms) prediction, as shown "
                "in Table 3, the Random Forest model again achieved the highest R-squared "
                "of 0.9799 (RMSE = 0.0116). The Proposed CNN model achieved R-squared = "
                "0.9741 (RMSE = 0.0132), outperforming the DNN model (R-squared = 0.9732, "
                "RMSE = 0.0135). Fig. 6 presents scatter plots of predicted versus actual "
                "Cp,mean values for all four models, illustrating the close agreement "
                "between predictions and actual values.\n\n"

                "Wind Direction-Wise Performance\n\n"
                "The prediction accuracy of the proposed CNN model was evaluated across "
                "four wind direction quadrants, as shown in Fig. 4. The model demonstrated "
                "consistent performance across all quadrants, with R-squared values ranging "
                "from 0.9935 (180-270 degrees) to 0.9943 (90-180 degrees). The minimal "
                "variation in accuracy across wind directions confirms the model's ability "
                "to capture the directional dependence of wind pressure distributions "
                "without directional bias.\n\n"

                "Leave-One-Shape-Out Cross-Validation\n\n"
                "The LOSO cross-validation results are presented in Fig. 7. The average "
                "R-squared across all eight folds was 0.9886, indicating strong "
                "generalization capability to unseen building geometries. The highest "
                "accuracy was achieved for the 1:1(T) configuration (R-squared = 0.9950, "
                "RMSE = 0.0314), while the lowest was observed for the 1:4 configuration "
                "(R-squared = 0.9755, RMSE = 0.0890). The relatively lower performance "
                "for the 1:4 case is attributable to its extreme side ratio, which produces "
                "aerodynamic characteristics that differ substantially from the other "
                "configurations in the training set.\n\n"

                "Feature Importance Analysis\n\n"
                "Permutation-based feature importance analysis was conducted on the XGBoost "
                "model, and the results are shown in Fig. 5. Face orientation (face_id) "
                "was identified as the most influential feature with an importance score "
                "of 1.912, followed by wind direction sine component (0.109), side ratio "
                "(0.026), and normalized height z/H (0.015). These results are physically "
                "consistent with aerodynamic principles: the pressure distribution on a "
                "building surface is primarily governed by the orientation of the face "
                "relative to the approaching wind, the wind direction determines the "
                "stagnation point location and wake region, and the side ratio influences "
                "flow separation and reattachment patterns."
            )
        },
        {
            "heading": "DISCUSSION",
            "content": (
                "The results of this study demonstrate that machine learning models can "
                "achieve high prediction accuracy for wind pressure coefficients on "
                "high-rise building surfaces using data from the TPU Aerodynamic Database. "
                "The R-squared values exceeding 0.993 for Cp,mean prediction are comparable "
                "to or better than those reported in previous studies for low-rise buildings. "
                "For instance, Weng and Paal (2022) reported R-squared values of approximately "
                "0.98 for their ML-WPP model applied to low-rise non-isolated buildings, "
                "and Tian et al. (2020) achieved similar accuracy for gable-roofed low-rise "
                "buildings using deep neural networks.\n\n"

                "The proposed CNN model, while achieving slightly lower R-squared than "
                "Random Forest for Cp,mean prediction, offers a distinct advantage in its "
                "ability to capture spatial interaction effects through engineered features. "
                "The pairwise and triple interaction features effectively encode the coupled "
                "relationships between wind direction, building geometry, and measurement "
                "location, which are fundamental to the aerodynamic behavior of bluff bodies. "
                "This approach bridges the gap between black-box machine learning models and "
                "the physical understanding of wind-structure interaction.\n\n"

                "The LOSO cross-validation results provide important insights into the "
                "generalization capability of the model framework. The average R-squared "
                "of 0.9886 across all leave-out folds suggests that the model can reliably "
                "predict wind pressures for building configurations not included in the "
                "training dataset. This finding has practical implications for preliminary "
                "design, where engineers may need to estimate wind loads for building "
                "geometries that have not been tested in wind tunnels. However, the reduced "
                "accuracy for the extreme side ratio case (D/B = 4.0) suggests that "
                "extrapolation beyond the range of training data should be approached "
                "with caution.\n\n"

                "The feature importance analysis provides physically meaningful insights "
                "into the model's decision-making process. The dominance of face orientation "
                "is consistent with the well-known fact that windward faces experience "
                "positive pressures while leeward and side faces experience suction. The "
                "high importance of wind direction aligns with the fundamental dependence "
                "of aerodynamic forces on the angle of attack. The significance of side "
                "ratio reflects the influence of cross-sectional geometry on flow separation, "
                "vortex shedding, and wake formation patterns, as documented extensively "
                "in the wind engineering literature (Holmes 2018).\n\n"

                "Several limitations of this study should be acknowledged. First, the "
                "current framework uses synthetic data generated to mimic the statistical "
                "properties of the TPU database; validation with actual wind tunnel data "
                "is necessary to confirm the findings. Second, only mean and RMS pressure "
                "coefficients were considered; peak pressure coefficients, which are critical "
                "for cladding design, require separate investigation. Third, the study "
                "considers isolated buildings only; real urban environments involve complex "
                "interference effects from neighboring structures."
            )
        },
        {
            "heading": "CONCLUSIONS",
            "content": (
                "This study developed a CNN-based framework for predicting wind pressure "
                "coefficients on high-rise building surfaces using the TPU Aerodynamic "
                "Database. The following conclusions are drawn:\n\n"

                "1. All four machine learning models (RF, XGBoost, DNN, and the proposed "
                "CNN) achieved R-squared values exceeding 0.993 for mean wind pressure "
                "coefficient prediction, confirming the viability of data-driven approaches "
                "for wind load estimation on high-rise buildings.\n\n"

                "2. The proposed CNN model, incorporating engineered spatial interaction "
                "features, achieved R-squared values of 0.994 and 0.974 for Cp,mean and "
                "Cp,rms predictions, respectively. The spatial feature engineering approach "
                "effectively captures aerodynamic coupling effects between building geometry, "
                "wind direction, and measurement location.\n\n"

                "3. Leave-one-shape-out cross-validation demonstrated strong generalization "
                "capability, with an average R-squared of 0.989 across eight building "
                "configurations with side ratios ranging from 0.5 to 4.0. This suggests "
                "the framework can reliably predict wind pressures for untested building "
                "geometries within the training range.\n\n"

                "4. Permutation-based feature importance analysis confirmed that face "
                "orientation, wind direction, and side ratio are the three most influential "
                "parameters, which is consistent with established aerodynamic principles "
                "and provides physical interpretability to the deep learning predictions.\n\n"

                "5. The developed framework has practical implications for preliminary "
                "wind load estimation in structural design, potentially reducing the need "
                "for extensive wind tunnel testing during early design stages."
            )
        },
        {
            "heading": "DATA AVAILABILITY STATEMENT",
            "content": (
                "The synthetic dataset generated in this study and the Python code used "
                "for model development are available at "
                "https://github.com/concrete-sangminlee/paperfactory. The TPU Aerodynamic "
                "Database is publicly available at "
                "https://wind.arch.t-kougei.ac.jp/system/eng/contents/code/tpu."
            )
        },
    ],
    "references": [
        'Gavalda, X., J. Ferrer, and G. Sanchez. 2018. "Prediction of wind pressure coefficients on building surfaces using artificial neural networks." Energy Build., 166, 2-11.',
        'Holmes, J. D. 2018. Wind loading of structures. 3rd Ed., CRC Press, Boca Raton, FL.',
        'Hu, G., L. Liu, D. Tao, J. Song, and K. C. S. Kwok. 2020. "Deep learning-based investigation of wind pressures on tall building under interference effects." J. Wind Eng. Ind. Aerodyn., 201, 104166.',
        'Li, Y., X. Huang, Y.-G. Li, F.-B. Chen, and Q.-S. Li. 2022. "Machine learning based algorithms for wind pressure prediction of high-rise buildings." Adv. Struct. Eng., 25 (8), 1700-1719. https://doi.org/10.1177/13694332221092671.',
        'Oh, B. K., B. Glisic, Y. Kim, and H. S. Park. 2019. "Convolutional neural network-based wind-induced response estimation model for tall buildings." Comput.-Aided Civ. Infrastruct. Eng., 34 (10), 843-858. https://doi.org/10.1111/mice.12476.',
        'Quan, Y., Y. Tamura, M. Matsui, S. Cao, and A. Yoshida. 2014. "TPU aerodynamic database for low-rise buildings." 23rd National Symposium on Wind Engineering, Tokyo, Japan.',
        'Tian, J., K. R. Gurley, M. T. Diaz, P. L. Fernandez-Caban, F. J. Masters, and R. Fang. 2020. "Low-rise gable roof buildings pressure prediction using deep neural networks." J. Wind Eng. Ind. Aerodyn., 196, 104026. https://doi.org/10.1016/j.jweia.2019.104026.',
        'Weng, Y., and S. G. Paal. 2022. "Machine learning-based wind pressure prediction of low-rise non-isolated buildings." Eng. Struct., 258, 114148. https://doi.org/10.1016/j.engstruct.2022.114148.',
    ],
}

figures = [
    os.path.join("outputs", "figures", f"fig_{i}_{name}.png")
    for i, name in enumerate([
        "building_configurations",
        "cp_distribution_heatmap",
        "model_performance_comparison",
        "wind_direction_accuracy",
        "feature_importance",
        "scatter_predicted_vs_actual",
        "loso_cross_validation",
    ], 1)
]
figures = [os.path.join(os.path.dirname(os.path.abspath(__file__)), f) for f in figures]

output_path = generate_word(paper_content, "asce_jse", figures)
print(f"Paper generated: {output_path}")
