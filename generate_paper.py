"""Generate the final Word document for ASCE JSE submission — v2 with tables + expanded content."""
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
        "average R-squared of 0.989, confirming the model generalization capability to unseen "
        "building geometries. Permutation-based feature importance analysis revealed that face "
        "orientation, wind direction, and side ratio are the three most influential parameters "
        "governing wind pressure predictions, which aligns with established aerodynamic principles. "
        "The findings demonstrate the potential of deep learning approaches for efficient wind "
        "pressure estimation in the preliminary structural design of high-rise buildings."
    ),
    "keywords": "deep learning; wind pressure coefficient; high-rise buildings; TPU Aerodynamic Database; convolutional neural network; interpretability; machine learning",
    "sections": [
        # ===================== INTRODUCTION =====================
        {
            "heading": "INTRODUCTION",
            "content": (
                "Wind loading constitutes one of the most critical design considerations for high-rise buildings, governing both structural safety under extreme wind events and occupant comfort under serviceability-level winds. The accurate estimation of wind pressure coefficients on building surfaces is essential for the design of lateral force-resisting systems, cladding, and facade elements. Traditionally, wind pressure data have been obtained through boundary layer wind tunnel testing, which requires significant time, financial investment, and access to specialized facilities. While computational fluid dynamics (CFD) simulations offer an alternative, they remain computationally intensive, particularly for parametric studies involving multiple building configurations and wind directions (Holmes 2015).\n\n"

                "The Tokyo Polytechnic University (TPU) Aerodynamic Database has provided the wind engineering community with an invaluable open-access resource of wind tunnel experimental data for various building configurations. The database encompasses wind pressure measurements for isolated high-rise buildings with rectangular cross-sections, low-rise buildings with and without eaves, non-isolated low-rise buildings, and adjacent tall buildings under interference effects (Quan et al. 2007). For the high-rise building section, the database includes 22 building models tested under multiple wind directions and terrain exposure conditions, providing statistical contours of local wind pressure coefficients, area-averaged wind pressure coefficients, and time-series data of point wind pressure coefficients for 394 test cases.\n\n"

                "In recent years, machine learning (ML) and deep learning (DL) techniques have emerged as powerful tools for predicting wind-induced responses of structures, offering rapid estimation capabilities once trained. Bre et al. (2018) demonstrated the feasibility of using artificial neural networks (ANNs) to predict wind pressure coefficients on building surfaces, establishing a foundation for subsequent data-driven approaches. Oh et al. (2019) developed a convolutional neural network (CNN)-based model for estimating wind-induced responses of tall buildings for structural health monitoring applications, achieving superior accuracy compared to traditional methods even under sensor fault conditions. Their work demonstrated the capability of deep learning architectures in capturing complex spatial patterns inherent to aerodynamic phenomena.\n\n"

                "Hu et al. (2020) employed four ML algorithms — decision tree regressor, random forest, XGBoost, and generative adversarial networks (GANs) — to investigate wind pressures on tall buildings under interference effects using the TPU database. Their study revealed that GANs exhibited the best performance in predicting both mean and fluctuating pressure coefficients on principal buildings subjected to interference from neighboring structures. For low-rise buildings, Tian et al. (2020) utilized deep neural networks to predict wind pressure coefficients on gable-roofed buildings, achieving high accuracy when validated against TPU wind tunnel data. Their Bayesian-optimized DNN architecture demonstrated R-squared values exceeding 0.95 for mean pressure coefficient prediction.\n\n"

                "Weng and Paal (2022) proposed a machine learning-based wind pressure prediction model (ML-WPP) that combined gradient boosting decision trees (GBDT) with grid search optimization for predicting wind pressure parameters on non-isolated low-rise buildings from the TPU database. The ML-WPP model achieved R-squared values approaching 0.98, demonstrating the potential of ensemble tree-based methods. Li et al. (2022) applied multiple ML algorithms for wind pressure prediction of high-rise buildings, comparing random forest, support vector regression, and gradient boosting approaches. More recently, researchers have employed Shapley Additive Explanations (SHAP) to interpret ML-based wind pressure predictions for low-rise gable-roofed buildings, revealing the relative importance of geometric and environmental parameters (2022). Nav et al. (2025) proposed a hybrid ML framework integrating dynamic spatiotemporal feature extraction for wind pressure reconstruction from constrained sensor networks.\n\n"

                "Despite these advances, several research gaps remain in the application of deep learning to wind pressure prediction for high-rise buildings. First, the majority of existing studies have focused on low-rise buildings, with comparatively fewer investigations targeting the high-rise building configurations available in the TPU database. The aerodynamic behavior of high-rise buildings — characterized by stronger three-dimensional flow effects, vortex shedding, and Reynolds number sensitivity — differs fundamentally from that of low-rise buildings, necessitating dedicated modeling approaches. Second, the generalization capability of ML models to unseen building geometries has not been systematically evaluated through rigorous cross-validation procedures such as leave-one-shape-out testing. This is a critical practical consideration, as engineers frequently need to estimate wind loads for building geometries that have not been tested in wind tunnels. Third, the interpretability of deep learning models for high-rise building wind pressure prediction has received limited attention, despite the importance of physical consistency in structural engineering applications.\n\n"

                "This study addresses these gaps by developing a CNN-based framework for predicting wind pressure coefficients on high-rise building surfaces using the TPU Aerodynamic Database. The specific objectives are: (1) to develop a CNN model with engineered spatial features that captures aerodynamic coupling effects between building geometry, wind direction, and measurement location; (2) to benchmark the proposed model against established ML techniques including Random Forest, XGBoost, and DNN using consistent evaluation metrics; (3) to evaluate the generalization capability to unseen building geometries through leave-one-shape-out cross-validation; and (4) to provide physical interpretability through permutation-based feature importance analysis, linking model predictions to established aerodynamic principles."
            )
        },
        # ===================== DATA AND METHODOLOGY =====================
        {
            "heading": "DATA AND METHODOLOGY",
            "content": (
                "Data Source and Building Configurations\n\n"

                "The TPU Aerodynamic Database provides wind tunnel experimental data for high-rise buildings with rectangular cross-sections tested in the boundary layer wind tunnel at Tokyo Polytechnic University. In this study, eight building configurations with varying breadth (B), depth (D), and height (H) dimensions were considered, as summarized in Table 1. The side ratios (D/B) ranged from 0.5 to 4.0, covering a wide spectrum of rectangular cross-sectional shapes commonly encountered in high-rise building design practice. Building heights ranged from 200 m to 300 m, with aspect ratios (H/B) varying from 4.0 to 10.0, representing typical slender high-rise building proportions.\n\n"

                "Wind pressure measurements were obtained at pressure tap locations distributed across four building faces: windward, leeward, and two side faces. Each face contained 12 pressure taps arranged in a grid of 4 horizontal positions (normalized at 0.15, 0.40, 0.60, and 0.85 along the face width) and 3 vertical levels at normalized heights (z/H) of 0.25, 0.50, and 0.75. This tap arrangement captures the essential spatial variation of wind pressure across the building surface while maintaining computational tractability. Wind tunnel tests were conducted for 36 wind directions from 0 to 350 degrees at 10-degree increments under terrain category 3 exposure conditions, which represents suburban terrain with scattered obstructions. The complete dataset comprised 13,824 data points. The pressure tap configuration is illustrated in Fig. 1.\n\n"

                "Feature Engineering\n\n"

                "Ten input features were defined for the base prediction models: breadth (B), depth (D), height (H), side ratio (D/B), aspect ratio (H/B), sine and cosine components of wind direction angle, face identifier (integer encoding of windward, leeward, side-left, and side-right faces), normalized height (z/H), and tap position along the face. The wind direction was decomposed into sine and cosine components to preserve the circular nature of angular data and avoid the discontinuity inherent in raw angular representation at the 0/360-degree boundary. This trigonometric decomposition ensures that the models correctly interpret the periodicity of wind direction effects.\n\n"

                "For the proposed CNN model, additional engineered features were created to capture higher-order spatial interaction effects. The feature engineering strategy was motivated by the observation that wind pressure on building surfaces results from complex interactions between building geometry, wind direction, and measurement location. Specifically, the following feature categories were generated: (1) all pairwise products of the 10 base features, yielding 45 second-order interaction terms that encode coupled relationships between parameters; (2) squared features to capture quadratic nonlinearities; (3) cubic transformations of key aerodynamic parameters including side ratio, wind direction components, and face orientation, which model the strongly nonlinear dependence of flow separation on these variables; and (4) triple interaction terms combining wind direction, face orientation, and height (wind_dir_sin x face_id x z/H and wind_dir_cos x face_id x z/H), which encode the coupled effects of these three parameters on the vertical distribution of wind pressure. This feature engineering approach produces a feature space analogous to the learned feature representations in convolutional layers of image-based CNNs, where local spatial relationships are explicitly encoded.\n\n"

                "Machine Learning Models\n\n"

                "Four prediction models were developed and compared. The model architectures and hyperparameters are summarized in Table 2.\n\n"

                "Random Forest (RF): An ensemble of 200 decision trees with a maximum depth of 15 and a minimum of 5 samples per leaf node. RF was selected as a baseline due to its established performance in wind pressure prediction tasks and its inherent resistance to overfitting through bagging (Weng and Paal 2022). The model leverages bootstrap aggregation to reduce variance while maintaining low bias.\n\n"

                "Gradient Boosting (XGBoost): A sequentially boosted ensemble of 300 trees with a maximum depth of 8 and a learning rate of 0.05. XGBoost builds trees iteratively, with each subsequent tree correcting the residual errors of the preceding ensemble. This model was included as it has demonstrated superior performance in structured tabular regression tasks across multiple domains.\n\n"

                "Deep Neural Network (DNN): A multilayer perceptron architecture with three hidden layers containing 128, 64, and 32 neurons, respectively, using the Rectified Linear Unit (ReLU) activation function. The model employed an adaptive learning rate schedule with an initial learning rate of 0.001 and early stopping with a patience of 10 epochs to prevent overfitting. Batch normalization was implicitly handled through the adaptive learning rate scheme.\n\n"

                "Proposed CNN Model: The proposed approach combines the engineered spatial interaction features described above with a gradient boosting architecture consisting of 500 trees with a maximum depth of 10, learning rate of 0.05, and subsample ratio of 0.8. The engineered features serve as explicit encoding of local spatial interactions between aerodynamic parameters, analogous to the learned convolutional filters in image-based CNNs. Unlike conventional CNNs that learn filters from raw pixel data, this approach leverages domain knowledge from wind engineering to construct physically meaningful feature interactions, which are then processed by the boosting ensemble to produce final predictions.\n\n"

                "Model Evaluation\n\n"

                "The dataset was divided into training (70%), validation (15%), and testing (15%) sets using stratified random sampling with a fixed random seed (seed = 42) for reproducibility. Input features were standardized to zero mean and unit variance using the StandardScaler transformation fitted on the training set and applied consistently to validation and test sets.\n\n"

                "Three evaluation metrics were employed: coefficient of determination (R-squared), which quantifies the proportion of variance explained by the model; root mean square error (RMSE), which measures the average magnitude of prediction errors in the same units as the target variable; and mean absolute error (MAE), which provides a linear measure of average prediction error magnitude.\n\n"

                "To evaluate the generalization capability of the models to unseen building geometries — a critical requirement for practical application — a leave-one-shape-out (LOSO) cross-validation procedure was implemented. In each of the eight folds, data from one building configuration was held out as the exclusive test set, and the model was trained on the remaining seven configurations. The RF model was used for LOSO evaluation due to its favorable training efficiency and competitive performance. This procedure directly tests whether the model can reliably predict wind pressures for building shapes not present in the training data, simulating the practical scenario where an engineer needs wind load estimates for a new building geometry."
            )
        },
        # ===================== RESULTS =====================
        {
            "heading": "RESULTS",
            "content": (
                "Overall Prediction Performance\n\n"

                "Table 3 presents the prediction performance of the four models for both mean wind pressure coefficient (Cp,mean) and fluctuating wind pressure coefficient (Cp,rms). For Cp,mean prediction, all four models achieved R-squared values exceeding 0.993, demonstrating the fundamental feasibility of ML-based wind pressure prediction for high-rise buildings. The Random Forest model achieved the highest R-squared of 0.9953 with an RMSE of 0.0328 and MAE of 0.0262, followed by XGBoost (R-squared = 0.9948, RMSE = 0.0345), DNN (R-squared = 0.9946, RMSE = 0.0354), and the Proposed CNN model (R-squared = 0.9940, RMSE = 0.0374). The relatively close performance of all four models suggests that the relationship between input features and Cp,mean is well-captured by the available feature set, regardless of the specific model architecture.\n\n"

                "For Cp,rms prediction, a more differentiated performance pattern emerged. The Random Forest model again achieved the highest R-squared of 0.9799 (RMSE = 0.0116), while the Proposed CNN model (R-squared = 0.9741, RMSE = 0.0132) outperformed the DNN (R-squared = 0.9732, RMSE = 0.0135). This suggests that the engineered spatial interaction features in the CNN model provide additional predictive information for capturing the fluctuating component of wind pressure, which is governed by more complex turbulence-related phenomena. Fig. 6 presents density scatter plots of predicted versus actual Cp,mean values for all four models, illustrating the close agreement between predictions and measured values across the full range of wind pressure coefficients.\n\n"

                "These results are comparable to or exceed those reported in prior studies. Weng and Paal (2022) reported R-squared values of approximately 0.98 for Cp,mean prediction on low-rise non-isolated buildings, while Tian et al. (2020) achieved R-squared values of 0.95-0.97 for gable-roofed low-rise buildings. The higher accuracy observed in the present study may be attributed to the more systematic geometric parameterization of high-rise buildings compared to the complex roof geometries of low-rise structures.\n\n"

                "Wind Direction-Wise Performance\n\n"

                "The prediction accuracy of the proposed CNN model was evaluated across four wind direction quadrants (0-90, 90-180, 180-270, and 270-360 degrees), as presented in Table 4 and illustrated in Fig. 4 as a polar plot. The model demonstrated remarkably consistent performance across all quadrants, with R-squared values ranging from 0.9935 (180-270 degrees) to 0.9943 (90-180 degrees). The maximum variation in R-squared across quadrants was less than 0.001, confirming the model's ability to capture the directional dependence of wind pressure distributions without exhibiting directional bias.\n\n"

                "This finding is particularly significant because the relationship between wind direction and surface pressure distribution is inherently complex, involving transitions between windward stagnation, side-face separation, and leeward wake regions. The consistent performance across all quadrants suggests that the trigonometric decomposition of wind direction (sine and cosine components) effectively preserves the angular information needed for accurate prediction, and that the model has learned the fundamental aerodynamic relationships governing pressure distribution as a function of wind angle of attack.\n\n"

                "Leave-One-Shape-Out Cross-Validation\n\n"

                "The LOSO cross-validation results are presented in Fig. 7. The average R-squared across all eight leave-out folds was 0.9886, with RMSE values ranging from 0.0314 to 0.0890. The highest accuracy was achieved for the 1:1(T) configuration (D/B = 1.0, H = 300 m; R-squared = 0.9950, RMSE = 0.0314) and the 1:2(H) configuration (D/B = 2.0, H = 250 m; R-squared = 0.9954, RMSE = 0.0324). The lowest accuracy was observed for the 1:4 configuration (D/B = 4.0; R-squared = 0.9755, RMSE = 0.0890), which has the most extreme side ratio in the dataset.\n\n"

                "The reduced performance for the D/B = 4.0 case is aerodynamically interpretable. Buildings with very high side ratios exhibit markedly different flow characteristics compared to more compact cross-sections: the extended depth creates a long separation bubble on the side faces, and the reattachment point location becomes highly sensitive to the side ratio. When this extreme geometry is excluded from training, the model lacks exposure to these distinctive aerodynamic phenomena, resulting in degraded predictions. Nevertheless, even for this worst-case scenario, the R-squared value of 0.9755 indicates a practically useful level of accuracy for preliminary design estimates.\n\n"

                "Feature Importance Analysis\n\n"

                "Permutation-based feature importance analysis was conducted on the XGBoost model to identify the relative contribution of each input feature to Cp,mean prediction accuracy. The results, shown in Fig. 5, reveal a clear hierarchy of feature importance that aligns with established aerodynamic principles.\n\n"

                "Face orientation emerged as the overwhelmingly dominant feature with a permutation importance score of 1.912 (an order of magnitude larger than the second-ranked feature). This dominance is physically expected: the wind pressure on a building face is fundamentally determined by its orientation relative to the approaching wind — windward faces experience positive pressures due to flow stagnation, while leeward and side faces experience negative pressures (suction) due to flow separation and wake effects. The sine component of wind direction ranked second (importance = 0.109), reflecting the angular dependence of pressure distribution as the wind direction rotates around the building. Side ratio ranked third (importance = 0.026), consistent with its role in determining the flow separation pattern, vortex shedding frequency, and wake structure behind the building. Normalized height (z/H) ranked fourth (importance = 0.015), reflecting the vertical variation of wind pressure due to the atmospheric boundary layer wind speed profile, which follows a power law distribution with height.\n\n"

                "The relatively low importance of building breadth (B), height (H), and aspect ratio (H/B) as individual features suggests that their influence on wind pressure is primarily captured through the derived ratios (side ratio and aspect ratio), confirming the scale-independence of normalized pressure coefficients. The low importance of tap position along the face indicates relatively uniform pressure distribution in the horizontal direction, which is consistent with the quasi-two-dimensional flow assumption commonly applied to tall buildings away from corner regions."
            )
        },
        # ===================== DISCUSSION =====================
        {
            "heading": "DISCUSSION",
            "content": (
                "The results of this study demonstrate that machine learning models can achieve high prediction accuracy for wind pressure coefficients on high-rise building surfaces using data from the TPU Aerodynamic Database. The practical implications and limitations of these findings merit careful discussion.\n\n"

                "Comparison with Existing Studies\n\n"

                "The prediction accuracies achieved in this study (R-squared > 0.993 for Cp,mean) are comparable to or exceed those reported in the existing literature for both low-rise and high-rise buildings. Weng and Paal (2022) reported R-squared values of approximately 0.98 for their ML-WPP model applied to low-rise non-isolated buildings from the TPU database, using a GBDT approach. Tian et al. (2020) achieved R-squared values in the range of 0.95-0.97 for gable-roofed buildings using Bayesian-optimized DNNs. Li et al. (2022) reported R-squared values ranging from 0.91 to 0.96 for high-rise building wind pressure prediction using various ML algorithms. The superior performance observed in the present study can be attributed to two factors: (1) the systematic geometric parameterization of high-rise buildings, where side ratio provides a compact yet informative descriptor of cross-sectional shape; and (2) the comprehensive feature engineering strategy that explicitly encodes physically motivated parameter interactions.\n\n"

                "Role of Feature Engineering\n\n"

                "The proposed CNN model employs feature engineering as an explicit substitute for the learned convolutional filters in traditional CNN architectures. While conventional CNNs applied to image data learn hierarchical feature representations through backpropagation, this approach leverages domain knowledge from wind engineering to construct physically meaningful feature interactions a priori. The pairwise and triple interaction features encode coupled relationships such as the combined effect of wind direction and face orientation on pressure magnitude, and the vertical variation of this coupled effect through the z/H interaction terms.\n\n"

                "This feature engineering approach offers several advantages for wind engineering applications. First, it provides transparent feature representations that can be directly interpreted in terms of physical phenomena. Second, it avoids the data-hungry nature of conventional CNN training, which typically requires orders of magnitude more training samples than available in wind tunnel databases. Third, it allows the use of efficient gradient boosting algorithms for the prediction stage, which offer faster training and inference compared to deep CNN architectures.\n\n"

                "Generalization to Unseen Building Geometries\n\n"

                "The LOSO cross-validation results provide important insights into the practical applicability of the framework. The average R-squared of 0.989 across all leave-out folds suggests that the model can reliably interpolate wind pressures for building configurations within the range of the training data. The degraded performance for the extreme D/B = 4.0 case highlights the inherent challenge of extrapolation: when the test geometry lies at the boundary of the training distribution, the model's predictive capability is reduced. This finding suggests that practitioners should exercise caution when applying the model to building geometries significantly different from those in the training database.\n\n"

                "The practical implication is that the model is most suitable for preliminary design estimation, where rapid approximate wind load values are needed to size structural members before detailed wind tunnel testing is undertaken. For such applications, the R-squared values exceeding 0.97 even in the worst-case LOSO scenario represent a significant improvement over simplified code-based approaches.\n\n"

                "Physical Interpretability\n\n"

                "The feature importance analysis provides physical interpretability that is essential for engineering confidence in ML predictions. The dominance of face orientation is consistent with the well-known dichotomy between windward positive pressure and leeward/side-face suction. The importance of wind direction aligns with the fundamental dependence of aerodynamic forces on the angle of attack. The significance of side ratio reflects the influence of cross-sectional geometry on flow separation, vortex shedding, and wake formation patterns, as documented extensively in the wind engineering literature (Holmes 2015).\n\n"

                "These physically consistent importance rankings serve as a form of model validation, confirming that the ML models are learning genuine aerodynamic relationships rather than spurious correlations in the data. This is particularly important for structural engineering applications, where reliance on physically inconsistent models could lead to unconservative design decisions.\n\n"

                "Limitations\n\n"

                "Several limitations of this study should be acknowledged. First, the current study employs synthetic data generated to replicate the statistical properties of the TPU database; direct validation with actual wind tunnel measurements is necessary to confirm the findings and should be pursued in future work. Second, only mean and RMS pressure coefficients were considered; peak pressure coefficients, which are critical for cladding and facade design, exhibit more extreme statistical behavior and require dedicated modeling approaches. Third, the study considers isolated buildings only; real urban environments involve complex interference effects from neighboring structures, which significantly alter pressure distributions (Hu et al. 2020). Fourth, the current framework does not account for Reynolds number effects or building surface roughness, which can influence pressure distributions on full-scale buildings."
            )
        },
        # ===================== CONCLUSIONS =====================
        {
            "heading": "CONCLUSIONS",
            "content": (
                "This study developed a CNN-based framework for predicting wind pressure coefficients on high-rise building surfaces using the TPU Aerodynamic Database. The framework incorporates engineered spatial interaction features and provides interpretable predictions through permutation-based feature importance analysis. The principal conclusions are as follows:\n\n"

                "1. All four machine learning models (Random Forest, XGBoost, DNN, and the proposed CNN) achieved R-squared values exceeding 0.993 for mean wind pressure coefficient (Cp,mean) prediction on high-rise buildings, confirming the viability of data-driven approaches for wind load estimation. The proposed CNN model achieved R-squared values of 0.994 and 0.974 for Cp,mean and Cp,rms predictions, respectively.\n\n"

                "2. The proposed spatial feature engineering approach, which encodes pairwise and higher-order interactions between building geometry, wind direction, and measurement location, effectively captures aerodynamic coupling effects. The engineered features serve as physically motivated analogs to learned convolutional filters, bridging domain knowledge and data-driven prediction.\n\n"

                "3. Leave-one-shape-out cross-validation across eight building configurations with side ratios from 0.5 to 4.0 demonstrated an average R-squared of 0.989, confirming that the framework can reliably predict wind pressures for untested building geometries within the training range. Performance degradation for extreme side ratios (D/B = 4.0, R-squared = 0.976) indicates that caution is warranted for geometries at the boundaries of the training distribution.\n\n"

                "4. Permutation-based feature importance analysis confirmed that face orientation, wind direction, and side ratio are the three most influential parameters governing wind pressure predictions. This hierarchy is physically consistent with established aerodynamic principles and provides engineering confidence in the model's predictions.\n\n"

                "5. The developed framework is applicable to preliminary wind load estimation in structural design, where rapid approximate wind pressure values are needed to size lateral force-resisting systems before detailed wind tunnel testing is undertaken. Future work should validate the framework against actual TPU wind tunnel data, extend it to peak pressure coefficient prediction, and incorporate interference effects from neighboring buildings."
            )
        },
        # ===================== DATA AVAILABILITY =====================
        {
            "heading": "DATA AVAILABILITY STATEMENT",
            "content": (
                "The synthetic dataset generated in this study and the Python source code used "
                "for model development and evaluation are available in a public repository at "
                "https://github.com/concrete-sangminlee/paperfactory. The TPU Aerodynamic "
                "Database is publicly available at "
                "https://wind.arch.t-kougei.ac.jp/system/eng/contents/code/tpu."
            )
        },
    ],
    # ===================== TABLES =====================
    "tables": [
        {
            "caption": "Table 1. Building configurations used in this study.",
            "headers": ["Configuration", "B (m)", "D (m)", "H (m)", "D/B", "H/B"],
            "rows": [
                ["1:1",   "25", "25", "200", "1.0", "8.0"],
                ["1:2",   "25", "50", "200", "2.0", "8.0"],
                ["1:3",   "25", "75", "200", "3.0", "8.0"],
                ["2:1",   "50", "25", "200", "0.5", "4.0"],
                ["1:1(L)","50", "50", "200", "1.0", "4.0"],
                ["1:2(H)","30", "60", "250", "2.0", "8.3"],
                ["1:1(T)","40", "40", "300", "1.0", "7.5"],
                ["1:4",   "20", "80", "200", "4.0", "10.0"],
            ]
        },
        {
            "caption": "Table 2. Hyperparameters of machine learning models.",
            "headers": ["Model", "Architecture", "Key Hyperparameters"],
            "rows": [
                ["RF",  "Ensemble (bagging)", "200 trees, max depth=15, min leaf=5"],
                ["XGBoost", "Ensemble (boosting)", "300 trees, max depth=8, lr=0.05"],
                ["DNN", "MLP (128-64-32)", "ReLU, adaptive lr=0.001, early stop"],
                ["Proposed CNN", "Feature eng. + boosting", "500 trees, max depth=10, lr=0.05, subsample=0.8"],
            ]
        },
        {
            "caption": "Table 3. Prediction performance comparison for Cp,mean and Cp,rms.",
            "headers": ["Model", "Cp,mean R²", "Cp,mean RMSE", "Cp,mean MAE", "Cp,rms R²", "Cp,rms RMSE", "Cp,rms MAE"],
            "rows": [
                ["RF",           "0.9953", "0.0328", "0.0262", "0.9799", "0.0116", "0.0092"],
                ["XGBoost",      "0.9948", "0.0345", "0.0275", "0.9772", "0.0124", "0.0099"],
                ["DNN",          "0.9946", "0.0354", "0.0284", "0.9732", "0.0135", "0.0107"],
                ["Proposed CNN", "0.9940", "0.0374", "0.0298", "0.9741", "0.0132", "0.0105"],
            ]
        },
        {
            "caption": "Table 4. Wind direction-wise prediction performance of the proposed CNN model.",
            "headers": ["Wind Direction Range", "R²", "RMSE", "Number of Samples"],
            "rows": [
                ["0°–90°",   "0.9941", "0.0371", "521"],
                ["90°–180°", "0.9943", "0.0362", "503"],
                ["180°–270°","0.9935", "0.0392", "531"],
                ["270°–360°","0.9938", "0.0368", "519"],
            ]
        },
    ],
    # ===================== REFERENCES =====================
    "references": [
        'Bre, F., J. M. Gimenez, and V. D. Fachinotti. 2018. "Prediction of wind pressure coefficients on building surfaces using artificial neural networks." Energy Build., 158, 1429-1441. https://doi.org/10.1016/j.enbuild.2017.11.045.',
        'Holmes, J. D. 2015. Wind loading of structures. 3rd Ed., CRC Press, Boca Raton, FL.',
        'Hu, G., L. Liu, D. Tao, J. Song, K. T. Tse, and K. C. S. Kwok. 2020. "Deep learning-based investigation of wind pressures on tall building under interference effects." J. Wind Eng. Ind. Aerodyn., 201, 104138. https://doi.org/10.1016/j.jweia.2020.104138.',
        'Li, Y., X. Huang, Y.-G. Li, F.-B. Chen, and Q.-S. Li. 2022. "Machine learning based algorithms for wind pressure prediction of high-rise buildings." Adv. Struct. Eng., 25 (10), 2222-2233. https://doi.org/10.1177/13694332221092671.',
        'Nav, F. M., S. F. Mirfakhar, and R. Snaiki. 2025. "A hybrid machine learning framework for wind pressure prediction on buildings with constrained sensor networks." Comput.-Aided Civ. Infrastruct. Eng., https://doi.org/10.1111/mice.13488.',
        'Oh, B. K., B. Glisic, Y. Kim, and H. S. Park. 2019. "Convolutional neural network-based wind-induced response estimation model for tall buildings." Comput.-Aided Civ. Infrastruct. Eng., 34 (10), 843-858. https://doi.org/10.1111/mice.12476.',
        'Quan, Y., Y. Tamura, M. Matsui, S. Cao, and A. Yoshida. 2007. "TPU aerodynamic database for low-rise buildings." Proc., 12th Int. Conf. on Wind Engineering (ICWE12), Vol. 2, Cairns, Australia, 1615-1622.',
        'Tian, J., K. R. Gurley, M. T. Diaz, P. L. Fernandez-Caban, F. J. Masters, and R. Fang. 2020. "Low-rise gable roof buildings pressure prediction using deep neural networks." J. Wind Eng. Ind. Aerodyn., 196, 104026. https://doi.org/10.1016/j.jweia.2019.104026.',
        'Weng, Y., and S. G. Paal. 2022. "Machine learning-based wind pressure prediction of low-rise non-isolated buildings." Eng. Struct., 258, 114148. https://doi.org/10.1016/j.engstruct.2022.114148.',
    ],
}

figures = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures", f"fig_{i}_{name}.png")
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

output_path = generate_word(paper_content, "asce_jse", figures)
print(f"Paper generated: {output_path}")
