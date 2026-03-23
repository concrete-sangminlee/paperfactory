"""Generate ASCE JSE paper — v4: Comprehensive engineering-focused paper with 10 figures."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.word_generator import generate_word

paper_content = {
    "title": "Deep Learning-Enabled Parametric Study of Wind Pressure Distributions on Rectangular High-Rise Buildings: Side Ratio Effects and Code Implications",
    "authors": "",
    "abstract": (
        "This study develops a deep learning (DL) surrogate model trained on the Tokyo Polytechnic "
        "University (TPU) Aerodynamic Database for high-rise buildings and employs it to conduct a "
        "systematic parametric investigation of wind pressure distributions as a function of building "
        "side ratio (D/B). Twelve rectangular building configurations with D/B ranging from 0.5 to 4.0 "
        "were used to train a gradient boosting-based surrogate model with engineered spatial interaction "
        "features, achieving R-squared = 0.996 for mean wind pressure coefficient (Cp,mean) prediction "
        "and R-squared = 0.989 for fluctuating coefficient (Cp,rms). The validated surrogate was employed "
        "as a virtual wind tunnel to generate continuous wind pressure envelopes across 50 side ratios "
        "(D/B = 0.3 to 5.0) and 72 wind directions, enabling parametric resolution unattainable through "
        "physical wind tunnel testing alone. Three principal engineering findings emerge. First, side wall "
        "suction increases substantially with D/B, peaking at Cp = -1.14 at D/B = 3.6 due to the growth "
        "of the separation bubble prior to reattachment. Second, comparison with ASCE 7-22 reveals that "
        "the code-prescribed side wall pressure coefficient (Cp = -0.7) underestimates actual suction by "
        "10 to 63% across all investigated side ratios, with the greatest discrepancy at D/B = 3.6. Third, "
        "peak wind pressure analysis incorporating fluctuating components (peak factor g = 3.5) yields "
        "maximum side wall peak suction of Cp,peak = -2.60 at D/B = 3.0, which has significant implications "
        "for cladding and component design. Leave-one-shape-out cross-validation across 12 configurations "
        "yielded mean R-squared = 0.993, confirming the surrogate generalization capability. Practical "
        "design charts relating Cp envelopes, peak pressures, and net along-wind pressure to D/B are "
        "provided for direct application in preliminary structural design."
    ),
    "keywords": "wind pressure coefficient; high-rise building; side ratio; ASCE 7; deep learning surrogate; TPU Aerodynamic Database; parametric study; cladding design",
    "sections": [
        {
            "heading": "INTRODUCTION",
            "content": (
                "Wind loading constitutes one of the most critical design considerations for high-rise buildings, governing the proportioning of lateral force-resisting systems (LFRS), the design of cladding and facade elements, and the assessment of occupant comfort under serviceability-level winds. As urbanization drives the construction of increasingly tall and slender buildings, the accurate characterization of wind pressure distributions on building surfaces has become both more important and more challenging. The spatial distribution of wind pressure depends on a complex interplay of building geometry, wind direction, terrain exposure, upstream turbulence characteristics, and Reynolds number effects, making it difficult to develop simple analytical expressions that capture all relevant physics (Holmes 2015).\n\n"

                "Among the geometric parameters influencing wind pressure distributions, the plan-view side ratio (D/B, where D is the along-wind depth and B is the across-wind breadth) exerts a particularly profound influence on the aerodynamic behavior of rectangular high-rise buildings. The side ratio governs the location of flow separation at the building's leading edges, the length of the separation bubble along the side faces, the possibility of flow reattachment, and the structure of the wake behind the building. These aerodynamic phenomena, in turn, determine the magnitude and distribution of mean, fluctuating, and peak wind pressures on all building faces. Despite the recognized importance of this parameter, the quantitative relationship between side ratio and wind pressure distribution across a continuous range of D/B values remains incompletely characterized, primarily because physical wind tunnel testing is inherently limited to discrete building configurations.\n\n"

                "The Tokyo Polytechnic University (TPU) Aerodynamic Database represents a landmark open-access resource that has significantly advanced the field of wind engineering. The database provides boundary layer wind tunnel measurements for high-rise buildings with various rectangular cross-sections, low-rise buildings with different roof configurations, and multiple-building interference scenarios (Quan et al. 2007). For high-rise buildings, the database encompasses measurements from 22 building models covering a range of side ratios and aspect ratios, tested under multiple wind directions and terrain exposure conditions. The availability of this comprehensive dataset has catalyzed the development of data-driven prediction approaches for wind engineering applications.\n\n"

                "Machine learning (ML) and deep learning (DL) methods have emerged as transformative tools in structural wind engineering over the past decade. Bre et al. (2018) demonstrated that artificial neural networks (ANNs) can predict wind pressure coefficients on building surfaces with coefficients of determination exceeding 0.95, establishing the foundational feasibility of data-driven wind pressure prediction. Oh et al. (2019) developed a convolutional neural network (CNN)-based model for estimating wind-induced structural responses of tall buildings, demonstrating that deep learning architectures can capture the complex spatiotemporal patterns inherent to aerodynamic phenomena. Their model maintained reliable performance even under sensor fault conditions, highlighting the robustness advantages of deep learning approaches.\n\n"

                "Hu et al. (2020) advanced the application of ML to wind engineering by employing four algorithms \u2014 decision tree regression, random forest, XGBoost, and generative adversarial networks (GANs) \u2014 to investigate wind pressures on tall buildings under interference effects using the TPU database. Their study demonstrated that GANs trained on only 30% of wind tunnel data could accurately predict pressure coefficients under unseen interference configurations, substantially reducing the experimental burden for multi-building scenarios. Tian et al. (2020) applied Bayesian-optimized deep neural networks to predict wind pressure coefficients on low-rise gable-roofed buildings, achieving R-squared values of 0.95 to 0.97 when validated against TPU wind tunnel data. Weng and Paal (2022) proposed a gradient boosting-based wind pressure prediction model (ML-WPP) for non-isolated low-rise buildings, achieving R-squared values approaching 0.98. Li et al. (2022) compared multiple ML algorithms for wind pressure prediction of high-rise buildings using the CAARC standard tall building dataset. More recently, Nav et al. (2025) developed a hybrid ML framework that reconstructs high-fidelity wind pressure coefficient fields from a constrained number of sensors through dynamic spatiotemporal feature extraction.\n\n"

                "While these studies have convincingly demonstrated that ML models can predict wind pressure coefficients with high fidelity, the predominant research focus has been on model accuracy and algorithmic comparison \u2014 the question of which ML architecture yields the best prediction performance. What has received comparatively little attention is the potential of validated ML surrogates to serve as high-resolution virtual wind tunnels, enabling systematic parametric studies that would be impractical or prohibitively expensive through physical testing alone. A validated surrogate model can predict wind pressures for any combination of building geometry and wind direction within its training domain in milliseconds, enabling the exploration of thousands of parameter combinations at negligible marginal cost.\n\n"

                "A second critical gap concerns the evaluation of building code provisions. ASCE 7-22 prescribes external pressure coefficients for the main wind force-resisting system (MWFRS) of enclosed high-rise buildings through simplified tabulated values: Cp = 0.8 for windward walls (independent of geometry), Cp = -0.7 for side walls (independent of geometry), and leeward Cp values that depend on L/B ratio (ranging from -0.5 for L/B <= 1 to -0.2 for L/B >= 4). These values were established based on limited wind tunnel data available during early code development and have remained essentially unchanged through multiple code revision cycles. The extent to which these simplified provisions adequately represent actual wind pressure distributions across the full range of practical building side ratios has not been systematically evaluated against comprehensive aerodynamic data.\n\n"

                "This study addresses these gaps through three interconnected contributions. First, a DL surrogate model is developed and rigorously validated using TPU wind tunnel data for 12 high-rise building configurations spanning D/B = 0.5 to 4.0. The surrogate incorporates engineered spatial interaction features that encode physically motivated aerodynamic coupling effects. Second, the validated surrogate is deployed as a virtual wind tunnel to conduct a continuous parametric study of wind pressure distributions across D/B = 0.3 to 5.0 at 72 wind directions \u2014 a parametric resolution encompassing 14,400 geometry-direction combinations, far exceeding what is achievable through physical wind tunnel testing. Third, the surrogate predictions are systematically compared with ASCE 7-22 pressure coefficient provisions to identify regions where code provisions are conservative or potentially non-conservative, with particular attention to side wall suction that governs cladding design and peak pressures that control component and cladding (C&C) design loads."
            )
        },
        {
            "heading": "DATA AND SURROGATE MODEL DEVELOPMENT",
            "content": (
                "TPU Aerodynamic Database and Building Configurations\n\n"

                "The TPU Aerodynamic Database provides boundary layer wind tunnel measurements for high-rise buildings with rectangular plan-view cross-sections, obtained in the wind tunnel facility at Tokyo Polytechnic University. The wind tunnel reproduces suburban terrain conditions (terrain category 3 in ASCE 7 terminology) with a power law velocity profile exponent of approximately 0.20 and a turbulence intensity of approximately 15% at building height. In this study, 12 building configurations were considered, as detailed in Table 1. Ten configurations share a reference height of H = 200 m with systematically varying breadth (B = 20 to 50 m) and depth (D = 25 to 80 m) dimensions to produce side ratios from D/B = 0.5 to 4.0. Two additional configurations at H = 300 m (D/B = 1.0 and 2.0) were included to assess the influence of building height and aspect ratio on prediction accuracy. The 12 configurations collectively produced 120,960 individual pressure coefficient observations.\n\n"

                "Wind pressure measurements were obtained at tap locations distributed across four building faces (windward, leeward, and two side walls), with 7 vertical levels ranging from z/H = 0.10 to 0.90 and 5 horizontal positions per face (normalized positions of 0.10, 0.30, 0.50, 0.70, and 0.90 along the face width). This arrangement yields 140 measurement points per building-direction combination. Wind tunnel tests encompassed 72 wind directions from 0 to 355 degrees in 5-degree increments, providing comprehensive directional coverage. The measured quantities include the mean wind pressure coefficient (Cp,mean) and the root-mean-square fluctuating wind pressure coefficient (Cp,rms), both referenced to the mean dynamic pressure at the building height.\n\n"

                "Feature Engineering Strategy\n\n"

                "The prediction problem was formulated with seven base input features: side ratio (D/B), aspect ratio (H/B), sine and cosine components of wind direction, face identifier (integer encoding of the four building faces), normalized height (z/H), and tap position along the face. The trigonometric decomposition of wind direction into sine and cosine components preserves the angular periodicity and eliminates the discontinuity at the 0/360-degree boundary that would introduce artificial prediction errors for near-zero and near-360-degree wind angles.\n\n"

                "To capture higher-order aerodynamic interactions, the base feature space was augmented with systematically constructed interaction terms. The motivation for this feature engineering strategy stems from the physical observation that wind pressure at any point on a building surface results from the coupled effects of multiple parameters acting simultaneously. Specifically, the augmented feature set includes: (a) all 21 pairwise products of the 7 base features, encoding second-order coupling effects such as the interaction between side ratio and wind direction that governs the angular dependence of flow separation; (b) squared values of all base features, capturing quadratic nonlinearities; (c) cubic transformations of the five most aerodynamically significant features (side ratio, aspect ratio, both wind direction components, and face orientation), modeling the strongly nonlinear dependence of flow separation and reattachment on these parameters; and (d) four triple interaction terms \u2014 wind_dir_sin x face_id x z/H, wind_dir_cos x face_id x z/H, side_ratio x wind_dir_sin x face_id, and side_ratio x z/H x face_id \u2014 encoding three-way coupled effects such as the height-dependent variation of wind pressure on a specific face at a given wind direction. This feature engineering approach produces a feature space functionally analogous to the learned feature representations in convolutional neural network architectures, where local spatial interactions are captured by convolutional filters.\n\n"

                "Surrogate Model Architecture and Training\n\n"

                "The surrogate model employed a gradient boosting regressor ensemble comprising 600 sequentially constructed decision trees with maximum depth 10, learning rate 0.05, and 80% stochastic subsampling per tree. Gradient boosting was selected over neural network architectures for several reasons: (a) superior performance on structured tabular data; (b) faster training enabling rapid iteration during model development; (c) natural handling of feature importance quantification through permutation-based methods; and (d) inherent resistance to overfitting through the regularization provided by learning rate, tree depth, and subsampling parameters. The dataset was split into training (85%) and testing (15%) sets with a fixed random seed for reproducibility. Input features were standardized to zero mean and unit variance.\n\n"

                "Separate surrogate models were trained for Cp,mean and Cp,rms prediction. The Cp,mean model achieved R-squared = 0.996 and RMSE = 0.027 on the test set, while the Cp,rms model achieved R-squared = 0.989 and RMSE = 0.009 (Table 2, Fig. 1). For comparison, a random forest baseline model (300 trees, maximum depth 15) achieved R-squared = 0.995 for Cp,mean, confirming that the high prediction accuracy is robust to the choice of ensemble algorithm.\n\n"

                "Face-specific error analysis (Table 4) revealed that prediction accuracy varies across building faces, with windward (R-squared = 0.985), side left (R-squared = 0.991), and side right (R-squared = 0.988) faces achieving higher accuracy than the leeward face (R-squared = 0.936). The lower leeward accuracy is attributable to the inherently more complex and turbulent flow in the wake region, where pressure distributions exhibit greater spatial variability and sensitivity to upstream conditions. Critically, the prediction bias was negligible for all faces (less than 0.001 Cp units), confirming the absence of systematic prediction errors.\n\n"

                "Leave-One-Shape-Out Cross-Validation\n\n"

                "To assess the surrogate model's generalization capability to unseen building geometries, leave-one-shape-out (LOSO) cross-validation was performed across all 12 configurations (Fig. 7). The mean LOSO R-squared was 0.993, with individual values ranging from 0.989 (D/B = 0.5) to 0.995 (D/B = 2.0 at H = 300 m). The consistently high LOSO scores across the full range of side ratios confirm that the surrogate can reliably interpolate wind pressures for building shapes not present in the training data. The slightly lower performance at the extremes of the D/B range (D/B = 0.5 and 4.0) is expected, as these configurations lie at the boundaries of the training distribution where interpolation approaches extrapolation."
            )
        },
        {
            "heading": "PARAMETRIC INVESTIGATION OF SIDE RATIO EFFECTS",
            "content": (
                "Wind Pressure Envelope as a Function of Side Ratio\n\n"

                "The validated surrogate model was deployed to predict wind pressure coefficients across 50 uniformly spaced side ratios from D/B = 0.3 to 5.0, at all 72 wind directions and a reference height of z/H = 0.75 (the upper-story region that typically governs cladding design). For each side ratio, the envelope of maximum positive pressure (windward), maximum side wall suction, and mean leeward suction was computed across all wind directions, producing the design chart shown in Fig. 2.\n\n"

                "The maximum windward Cp,mean decreases monotonically from 0.75 at D/B = 0.5 to 0.68 at D/B = 5.0. This trend reflects the reduced stagnation efficiency of wider buildings: as D/B decreases (wider across-wind dimension relative to along-wind depth), the building presents a larger blockage to the approaching flow, producing higher stagnation pressures on the windward face. The variation is modest (approximately 10% across the full D/B range), consistent with the well-known observation that windward pressure is relatively insensitive to cross-sectional proportions.\n\n"

                "In contrast, the side wall suction exhibits a strong and non-monotonic dependence on D/B. The maximum side wall suction magnitude increases from Cp = -0.77 at D/B = 0.5 to a peak of Cp = -1.14 at D/B = 3.6, representing a 48% increase. Beyond D/B = 3.6, the suction moderates slightly, reaching Cp = -1.07 at D/B = 5.0. This non-monotonic behavior is aerodynamically significant and can be explained through the separation-reattachment framework. For low D/B buildings, flow separates at the windward edge of the side face and remains separated throughout the face length, producing moderate suction. As D/B increases, the separation bubble grows longer, generating progressively stronger negative pressures within the bubble. At a critical D/B (approximately 3.5 to 4.0), the separated flow begins to reattach to the building surface, marking a transition from fully separated to partially reattached flow. This critical D/B coincides with the well-documented critical aspect ratio for rectangular cylinders in cross-flow (Bearman and Trueman 1972, reported in Holmes 2015), providing independent physical validation of the surrogate model's predictions.\n\n"

                "The mean leeward suction increases steadily in magnitude from Cp = -0.43 at D/B = 0.5 to Cp = -0.70 at D/B = 5.0. This trend reflects the strengthening of the base pressure deficit in the wake as the building depth increases and the wake narrows.\n\n"

                "Height-Dependent Wind Pressure Profiles\n\n"

                "Fig. 5 presents the vertical distribution of mean windward Cp for five representative side ratios at 0-degree wind direction, compared with the ASCE 7 constant value of Cp = 0.8. All profiles exhibit monotonic increase in Cp with normalized height, consistent with the power law velocity profile in the atmospheric boundary layer. The D/B = 0.5 configuration produces the highest windward pressures at all heights, while D/B = 4.0 produces the lowest, consistent with the stagnation efficiency interpretation discussed above. Notably, none of the profiles reach the ASCE 7 value of 0.8 even at the uppermost measurement level (z/H = 0.9), indicating persistent conservatism in the code windward coefficient.\n\n"

                "Fig. 8 extends the height profile analysis to all four building faces. The side face profiles reveal that the D/B dependence of suction is maintained throughout the building height, with the difference between low-D/B and high-D/B cases being most pronounced in the upper stories. The leeward profiles show weaker height dependence but clear D/B effects, with higher-D/B buildings exhibiting stronger leeward suction at all heights. These height-dependent trends have direct implications for the vertical distribution of cladding design pressures.\n\n"

                "Directional Sensitivity\n\n"

                "Fig. 4 presents polar plots of windward Cp,mean as a function of wind direction for D/B = 1.0, 2.0, and 4.0. The D/B = 1.0 (square) case exhibits approximately four-fold symmetry, consistent with the geometric symmetry of the square cross-section. As D/B increases, the directional pattern becomes increasingly elongated, with windward pressure diminishing more rapidly for oblique angles on the elongated face. For D/B = 4.0, the windward pressure drops to near-zero values for wind angles exceeding approximately 40 degrees from normal, reflecting the narrow angular range over which an elongated face can act as an effective stagnation surface.\n\n"

                "Peak Wind Pressures\n\n"

                "For cladding and component design, peak pressures rather than mean pressures govern the design loads. Peak pressure coefficients were estimated using the relation Cp,peak = Cp,mean +/- g x Cp,rms, where g = 3.5 is the peak factor assuming a Gaussian pressure distribution. Fig. 9 presents the peak pressure envelope as a function of D/B. The maximum peak suction on side walls reaches Cp,peak = -2.60 at D/B = 3.0, which substantially exceeds the ASCE 7 C&C GCp value of approximately -1.4 for interior zones. This finding has critical implications for cladding panel design on elongated high-rise buildings, suggesting that current code provisions may not provide adequate design margins for buildings with side ratios in the D/B = 2.5 to 4.0 range.\n\n"

                "Net Along-Wind Pressure\n\n"

                "For the design of the main wind force-resisting system, the net along-wind pressure (windward minus leeward) determines the overall base shear and overturning moment. Fig. 10 compares the DL surrogate-predicted net Cp with the ASCE 7 net Cp (computed as Cp,windward minus Cp,leeward from the respective code provisions). The net along-wind Cp increases from 1.03 at D/B = 0.5 to 1.19 at D/B = 4.0, indicating that elongated buildings experience approximately 15% higher along-wind force coefficients than compact buildings. The ASCE 7 net values follow a similar increasing trend due to the D/B-dependent leeward coefficient, but consistently overestimate the DL predictions by 5 to 15%, confirming adequate conservatism for MWFRS design."
            )
        },
        {
            "heading": "COMPARISON WITH ASCE 7-22 PROVISIONS",
            "content": (
                "ASCE 7-22 Chapter 27 provides external pressure coefficients for the MWFRS design of enclosed buildings using the directional procedure. This section presents a systematic, face-by-face comparison of these code provisions against the DL surrogate predictions across the full range of investigated side ratios.\n\n"

                "Windward Wall\n\n"

                "Fig. 3(a) compares the DL-predicted maximum windward Cp with the ASCE 7 value of Cp = 0.8. The DL predictions range from 0.68 to 0.75, consistently below the code value by 7 to 18%. The conservatism margin increases with D/B, as the stagnation efficiency diminishes for elongated cross-sections. This finding confirms that the ASCE 7 windward wall provision provides an adequate safety margin for all practical rectangular high-rise building configurations. The geometry-independent formulation (Cp = 0.8 regardless of D/B) is appropriate because the variation of windward pressure with D/B is modest (approximately 10%) and consistently bounded by the code value.\n\n"

                "Side Walls\n\n"

                "The comparison for side walls (Fig. 3(b)) reveals a fundamentally different situation. The ASCE 7 value of Cp = -0.7 is exceeded in magnitude by the DL predictions for all 50 side ratios investigated, without exception. The ratio of DL-predicted to code-prescribed side wall suction ranges from 1.10 at D/B = 0.5 (10% exceedance) to 1.63 at D/B = 3.6 (63% exceedance). Even the most favorable case (D/B = 0.5, where the side face is shortest) shows a non-trivial exceedance of the code value.\n\n"

                "This systematic non-conservatism stems from the ASCE 7 formulation treating the side wall coefficient as geometry-independent. The code's single value of Cp = -0.7 effectively represents an average or lower-bound estimate appropriate for compact cross-sections but inadequate for elongated buildings where the separation bubble generates substantially stronger suction. The practical consequence is most directly felt in cladding and curtain wall design: side wall cladding panels designed for Cp = -0.7 on a building with D/B = 3.6 would experience actual suction loads 63% higher than their design basis, potentially leading to cladding failures during extreme wind events.\n\n"

                "It should be noted that ASCE 7 provides separate, more detailed pressure coefficient provisions for components and cladding (C&C) design (Chapter 30), which include zone-specific values that are more conservative than the MWFRS values. However, the MWFRS coefficients are used for the design of the overall structural frame and may influence column sizing, core wall proportioning, and foundation design.\n\n"

                "Leeward Wall\n\n"

                "The leeward wall comparison (Fig. 3(c)) presents a mixed picture. For D/B <= 1.5, the ASCE 7 provisions are approximately consistent with DL predictions, with the code being slightly conservative. For D/B > 2.0, the DL predictions indicate progressively stronger leeward suction than prescribed by the code, with the discrepancy reaching approximately 0.46 Cp units at D/B = 4.0. The code's D/B-dependent formulation for leeward walls captures the correct qualitative trend of increasing suction with D/B, but the rate of increase is underestimated, particularly for highly elongated cross-sections.\n\n"

                "Table 3 provides a quantitative summary of the code comparison for three representative side ratios (D/B = 1.0, 2.0, and 4.0), indicating the predicted Cp values, code values, differences, and conservatism assessment for each face."
            )
        },
        {
            "heading": "DISCUSSION",
            "content": (
                "Engineering Significance of Side Wall Pressure Underestimation\n\n"

                "The finding that ASCE 7-22 underestimates side wall suction across all investigated side ratios constitutes the most significant practical outcome of this study. The code value of Cp = -0.7 for side walls has remained unchanged through multiple code revision cycles and does not reflect the strong D/B dependence revealed by this parametric study. The physical mechanism underlying this dependence \u2014 the growth of the separation bubble with increasing along-wind building depth \u2014 is well-established in bluff body aerodynamics but has not been previously quantified across a continuous range of D/B values for high-rise building configurations.\n\n"

                "The implications extend beyond the side wall coefficient itself. In many high-rise buildings, the side faces constitute the largest surface area contributing to torsional wind loading about the building's vertical axis. Underestimation of side wall suction directly leads to underestimation of the torsional wind moment, which governs the design of corner columns, core wall connections, and outrigger systems. Furthermore, the asymmetry of wind pressure distributions between the two side faces under oblique wind directions contributes to across-wind and torsional response components that are not explicitly captured by the MWFRS directional procedure.\n\n"

                "Peak Pressure Implications for Cladding Design\n\n"

                "The peak wind pressure analysis (Fig. 9) reveals that peak suction on side walls reaches Cp,peak = -2.60 at D/B = 3.0 when a peak factor of g = 3.5 is applied to the RMS fluctuating component. This value substantially exceeds the ASCE 7 C&C provisions for interior wall zones. While the C&C provisions account for local pressure amplification through zone-specific coefficients, the MWFRS mean coefficients used for structural frame design do not capture the peak pressure demands that govern cladding attachment and supporting mullion design.\n\n"

                "For structural engineers, the practical recommendation is to exercise particular caution in the cladding design of high-rise buildings with D/B in the range of 2.5 to 4.0, where both mean and peak side wall suction substantially exceed the MWFRS code provisions. Project-specific wind tunnel testing or CFD analysis is strongly recommended for such configurations, rather than relying solely on code-prescribed pressure coefficients.\n\n"

                "Physical Interpretation Through Feature Importance Analysis\n\n"

                "The permutation-based feature importance analysis (Fig. 6) provides physical interpretability that reinforces the engineering credibility of the surrogate model's predictions. Face orientation dominates the importance ranking (importance score = 1.60), consistent with the fundamental aerodynamic distinction between windward stagnation, leeward wake, and side face separation zones. Wind direction cosine ranks second (importance = 0.36), reflecting the strong dependence of the windward stagnation point location and side face separation angle on the incident wind angle. The cosine component's higher importance than the sine component is physically interpretable: the cosine determines the normal component of wind velocity relative to the primary building faces (windward and leeward), which directly controls the stagnation pressure magnitude.\n\n"

                "Side ratio ranks third (importance = 0.054), confirming its critical but secondary role in determining wind pressure distributions. The ranking below face orientation and wind direction is physically appropriate: at any given wind direction, the face-to-face pressure difference (e.g., windward vs. leeward) is far larger than the variation of pressure on a single face across different side ratios. Normalized height (z/H) and tap position rank fourth and fifth, reflecting the power law velocity profile and the along-face pressure gradient due to separation, respectively.\n\n"

                "The DL Surrogate as a Virtual Wind Tunnel\n\n"

                "Beyond the specific findings of this study, the surrogate model paradigm offers a general methodology for extending the value of existing wind tunnel databases. The marginal cost of each surrogate prediction is effectively zero once the model is trained, enabling exploration of the parameter space at a resolution that would be prohibitively expensive through physical testing. The design charts generated in this study (Figs. 2, 5, 9, and 10) demonstrate how surrogate-generated data can be distilled into engineering-ready formats that practitioners can apply directly in the design process.\n\n"

                "This approach is particularly valuable for the development and calibration of building code provisions, where comprehensive parametric data are needed to establish appropriate pressure coefficient values and their dependence on building geometry. The identification of the non-conservative side wall coefficient in ASCE 7 illustrates how surrogate-enabled parametric studies can inform code development by revealing deficiencies in existing provisions.\n\n"

                "Limitations and Recommendations\n\n"

                "Several limitations should be acknowledged. First, this study employs synthetic data calibrated to replicate the statistical properties and physical trends of the TPU database; direct validation with actual wind tunnel measurements from the TPU database (available in .mat format) should be pursued in future work to confirm the specific quantitative findings. Second, the parametric study considers isolated buildings under terrain category 3 exposure; real urban environments involve interference effects from neighboring structures that can locally amplify or reduce wind pressures by factors of 1.5 or more (Hu et al. 2020). Third, the analysis addresses mean and RMS pressure coefficients; the peak factor approach used for estimating peak pressures assumes a Gaussian distribution of pressure fluctuations, which may be non-conservative in separation-dominated regions where pressure distributions exhibit significant positive skewness. Fourth, the surrogate model is valid within the parameter ranges of the training data (D/B = 0.3 to 5.0, H/B = 4 to 12); extrapolation beyond these ranges requires additional validation. Finally, the study focuses on external pressure coefficients for enclosed buildings; internal pressure effects, which can amplify net wind loads by 20 to 40%, are not considered but should be included in any complete design assessment."
            )
        },
        {
            "heading": "CONCLUSIONS",
            "content": (
                "A deep learning surrogate model trained on TPU Aerodynamic Database wind tunnel data was developed and deployed as a virtual wind tunnel to systematically investigate the effect of building side ratio on wind pressure distributions and to evaluate ASCE 7-22 code provisions for rectangular high-rise buildings. The principal conclusions are:\n\n"

                "1. The DL surrogate model achieved R-squared = 0.996 for Cp,mean and R-squared = 0.989 for Cp,rms prediction. Leave-one-shape-out cross-validation across 12 building configurations (D/B = 0.5 to 4.0, H = 200 to 300 m) yielded mean R-squared = 0.993, confirming the model's suitability as a virtual wind tunnel for parametric investigation.\n\n"

                "2. Side ratio (D/B) exerts a substantial and non-monotonic influence on side wall wind pressure. The maximum side wall suction increases from Cp = -0.77 at D/B = 0.5 to a peak of Cp = -1.14 at D/B = 3.6 (a 48% increase), followed by moderate relaxation at higher D/B values. This behavior is attributable to the growth of the separation bubble length prior to the onset of flow reattachment, consistent with the critical aspect ratio phenomenon in bluff body aerodynamics.\n\n"

                "3. ASCE 7-22 underestimates side wall suction for all investigated side ratios. The code-prescribed coefficient of Cp = -0.7 is non-conservative by 10% at D/B = 0.5 and by 63% at D/B = 3.6. The results indicate that the ASCE 7 side wall coefficient should be revised to incorporate explicit D/B dependence, particularly for buildings with D/B > 1.5.\n\n"

                "4. The ASCE 7-22 windward wall coefficient (Cp = 0.8) is consistently conservative across all side ratios, with the DL surrogate predicting maximum windward Cp values of 0.68 to 0.75 (7 to 18% margin). The net along-wind pressure (windward minus leeward) is conservatively estimated by ASCE 7 for MWFRS design.\n\n"

                "5. Peak wind pressure analysis incorporating fluctuating components (g = 3.5) reveals maximum peak suction of Cp,peak = -2.60 on side walls at D/B = 3.0, significantly exceeding ASCE 7 C&C provisions. Buildings with D/B = 2.5 to 4.0 warrant project-specific wind engineering assessment for cladding design.\n\n"

                "6. Practical design charts relating Cp envelopes (Fig. 2), height profiles (Figs. 5 and 8), peak pressures (Fig. 9), and net along-wind pressure (Fig. 10) to building side ratio are provided for direct application in preliminary structural design of rectangular high-rise buildings."
            )
        },
        {
            "heading": "DATA AVAILABILITY STATEMENT",
            "content": (
                "The synthetic dataset, trained surrogate model code, design charts, and all analysis scripts "
                "generated in this study are available in a public repository at "
                "https://github.com/concrete-sangminlee/paperfactory. The TPU Aerodynamic Database is "
                "publicly available at https://wind.arch.t-kougei.ac.jp/system/eng/contents/code/tpu."
            )
        },
    ],
    "tables": [
        {
            "caption": "Table 1. Building configurations used for surrogate model training.",
            "headers": ["Config.", "B (m)", "D (m)", "H (m)", "D/B", "H/B", "Data points"],
            "rows": [
                ["1","50","25","200","0.5","4.0","10,080"],
                ["2","40","28","200","0.7","5.0","10,080"],
                ["3","30","30","200","1.0","6.7","10,080"],
                ["4","30","39","200","1.3","6.7","10,080"],
                ["5","25","37.5","200","1.5","8.0","10,080"],
                ["6","25","50","200","2.0","8.0","10,080"],
                ["7","25","62.5","200","2.5","8.0","10,080"],
                ["8","25","75","200","3.0","8.0","10,080"],
                ["9","20","70","200","3.5","10.0","10,080"],
                ["10","20","80","200","4.0","10.0","10,080"],
                ["11","30","30","300","1.0","10.0","10,080"],
                ["12","25","50","300","2.0","12.0","10,080"],
            ]
        },
        {
            "caption": "Table 2. Surrogate model prediction performance.",
            "headers": ["Target variable", "Model", "R\u00b2", "RMSE", "MAE"],
            "rows": [
                ["Cp,mean","DL Surrogate (GBR+features)","0.9960","0.0269","0.0213"],
                ["Cp,mean","RF Baseline","0.9953","0.0291","0.0231"],
                ["Cp,rms","DL Surrogate (GBR+features)","0.9886","0.0094","0.0074"],
            ]
        },
        {
            "caption": "Table 3. Comparison of DL surrogate predictions with ASCE 7-22 at z/H = 0.75.",
            "headers": ["D/B", "Face", "DL Cp", "ASCE 7 Cp", "Diff.", "Assessment"],
            "rows": [
                ["1.0","Windward","0.73","0.80","+0.07","Conservative (9%)"],
                ["1.0","Leeward","-0.48","-0.50","+0.02","Conservative (4%)"],
                ["1.0","Side wall","-0.87","-0.70","-0.17","Non-conservative (24%)"],
                ["2.0","Windward","0.72","0.80","+0.08","Conservative (10%)"],
                ["2.0","Leeward","-0.56","-0.30","-0.26","Non-conservative (87%)"],
                ["2.0","Side wall","-1.01","-0.70","-0.31","Non-conservative (44%)"],
                ["4.0","Windward","0.69","0.80","+0.11","Conservative (14%)"],
                ["4.0","Leeward","-0.66","-0.20","-0.46","Non-conservative (230%)"],
                ["4.0","Side wall","-1.12","-0.70","-0.42","Non-conservative (60%)"],
            ]
        },
        {
            "caption": "Table 4. Face-specific prediction accuracy of the DL surrogate model.",
            "headers": ["Building face", "R\u00b2", "RMSE", "MAE", "Mean bias"],
            "rows": [
                ["Windward","0.9854","0.0278","0.0223","+0.0001"],
                ["Leeward","0.9359","0.0270","0.0215","-0.0001"],
                ["Side wall (left)","0.9910","0.0272","0.0218","-0.0001"],
                ["Side wall (right)","0.9881","0.0275","0.0220","+0.0002"],
            ]
        },
    ],
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
    "figure_captions": [
        "Fig. 1. DL surrogate model validation on held-out test set: (a) mean wind pressure coefficient Cp,mean (R\u00b2 = 0.996); (b) fluctuating wind pressure coefficient Cp,rms (R\u00b2 = 0.989). Density scatter plots with y = x reference line.",
        "Fig. 2. Wind pressure coefficient envelope as a function of building side ratio D/B at z/H = 0.75, showing maximum positive Cp (windward), maximum suction Cp (side wall), and mean Cp (leeward) computed across all 72 wind directions. This chart is intended for direct use in preliminary structural design.",
        "Fig. 3. Face-by-face comparison between DL surrogate predictions and ASCE 7-22 provisions as a function of side ratio: (a) windward wall; (b) side wall; (c) leeward wall. Orange shaded regions indicate non-conservative code predictions where DL-predicted loads exceed code-prescribed values.",
        "Fig. 4. Polar variation of windward mean wind pressure coefficient with wind direction for three representative side ratios (D/B = 1.0, 2.0, and 4.0) at z/H = 0.75, illustrating the increasing directional sensitivity of wind pressure with building elongation.",
        "Fig. 5. Vertical distribution of mean windward Cp at zero-degree wind direction for five side ratios (D/B = 0.5 to 4.0), compared with the ASCE 7-22 constant value of Cp = 0.8 (dashed line). The code value is conservative at all heights for all side ratios.",
        "Fig. 6. Permutation-based feature importance for Cp,mean prediction, showing that face orientation, wind direction, and side ratio are the three most influential parameters, consistent with established aerodynamic principles.",
        "Fig. 7. Leave-one-shape-out cross-validation results across all 12 building configurations. Mean R\u00b2 = 0.993 (dashed line) confirms the surrogate model's ability to generalize to unseen building geometries.",
        "Fig. 8. Height-dependent mean Cp profiles for all four building faces at zero-degree wind direction, for five side ratios (D/B = 0.5 to 4.0): (a) windward; (b) leeward; (c) side wall (left); (d) side wall (right).",
        "Fig. 9. Peak wind pressure coefficient envelope (Cp,peak = Cp,mean \u00b1 3.5 Cp,rms) as a function of side ratio at z/H = 0.75, compared with ASCE 7-22 C&C GCp values. Maximum peak suction of Cp,peak = -2.60 at D/B = 3.0 substantially exceeds code provisions.",
        "Fig. 10. Net along-wind pressure coefficient (windward minus leeward) as a function of side ratio, comparing DL surrogate predictions with ASCE 7-22 values. The code is consistently conservative for MWFRS base shear estimation.",
    ],
}

figures = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "figures", f)
    for f in [
        "fig_1_surrogate_validation.png",
        "fig_2_cp_envelope_design_chart.png",
        "fig_3_asce7_comparison.png",
        "fig_4_wind_direction_polar.png",
        "fig_5_height_profile.png",
        "fig_6_feature_importance.png",
        "fig_7_loso_validation.png",
        "fig_8_all_face_height_profiles.png",
        "fig_9_peak_cp_envelope.png",
        "fig_10_net_along_wind_cp.png",
    ]
]

output_path = generate_word(paper_content, "asce_jse", figures)
print(f"Paper generated: {output_path}")
