###Streamlit应用程序开发
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('sub_model_mlp.pkl')
scaler = joblib.load('sub_scaler.pkl') 

# Streamlit user interface
st.title("Sirolimus Sub-therapeutic Risk Predictor")

# Define feature names
feature_names = ['BMI', 'WBC', 'MCH', 'TG', 'TBIL']

# 创建两列布局
col1, col2 = st.columns([1, 1.5])  # 左列稍窄，右列稍宽

with col1:
    # 输入表单
    with st.form("input_form"):
        BMI = st.number_input("BMI (kg/m2):", min_value=10.00, max_value=30.00, value=15.00)
        WBC = st.number_input("WBC (109/L):", min_value=0.00, max_value=20.00, value=8.00)
        MCH = st.number_input("MCH (pg):", min_value=10.00, max_value=50.00, value=25.00)
        TG = st.number_input("TG (mmol/L):", min_value=0.00, max_value=10.00, value=1.00)
        TBIL = st.number_input("TBIL (μmol/L):", min_value=0.00, max_value=20.00, value=5.00)
        submitted = st.form_submit_button("Predict")

# 准备输入特征
feature_values = [BMI, WBC, MCH, TG, TBIL]
features = np.array([feature_values])

# 关键修改：使用 pandas DataFrame 来确保列名
features_df = pd.DataFrame(features, columns=feature_names)
standardized_features_1 = scaler.transform(features_df)

# 关键修改：确保 final_features 是一个二维数组，并且用 DataFrame 传递给模型
final_features_df = pd.DataFrame(standardized_features_1, columns=feature_names)

if submitted: 
    with col1:
        # 这里可以留空或放一些其他内容
        pass

    with col2:
        OPTIMAL_THRESHOLD = 0.109
              
        predicted_proba = model.predict_proba(final_features_df)[0]
        prob_class1 = predicted_proba[1]  # 类别1的概率
    
        # 根据最优阈值判断类别
        predicted_class = 1 if prob_class1 >= OPTIMAL_THRESHOLD else 0

        # 先显示预测结果
        st.subheader("Prediction Results")

        # 使用更美观的方式显示结果
        if predicted_class == 1:
            st.error(f"Sub-therapeutic Risk: {prob_class1:.1%} (High Risk)")
        else:
            st.success(f"Sub-therapeutic Risk: {prob_class1:.1%} (Low Risk)") 
        
        st.write(f"**Risk Threshold:** {OPTIMAL_THRESHOLD:.0%} (optimized for clinical utility)")

        # 添加分隔线
        st.markdown("---")
        
        # 再显示SHAP解释图
        st.subheader("SHAP Explanation")


        # 创建SHAP解释器
        # 假设 X_train 是用于训练模型的特征数据
        df=pd.read_csv('Below_训练集_5变量.csv',encoding='utf8')
        trainy=df.Below
        x_train=df.drop('Below',axis=1)
        from sklearn.preprocessing import StandardScaler
        continuous_cols = ['BMI', 'WBC', 'MCH', 'TG', 'TBIL']
        trainx = x_train.copy()
        scaler = StandardScaler()
        trainx[continuous_cols] = scaler.fit_transform(x_train[continuous_cols])
    
        explainer_shap = shap.KernelExplainer(model.predict_proba, trainx)

        
        # 创建SHAP解释器
        ###explainer_shap = shap.KernelExplainer(model)
        ###explainer_shap = shap.KernelExplainer(model.predict_proba, trainx)

        # 获取SHAP值
        shap_values = explainer_shap.shap_values(pd.DataFrame(final_features_df,columns=feature_names))


        
        # 选择当前样本（假设第0个样本）和预测类别的SHAP值
        sample_idx = 0  # 你要解释的样本索引
        if predicted_class == 1:
            shap_values_single = shap_values[sample_idx, :, 1]  # (n_features,)
            expected_value = explainer_shap.expected_value[1]
        else:
            shap_values_single = shap_values[sample_idx, :, 0]  # (n_features,)
            expected_value = explainer_shap.expected_value[0]
        
        # 画瀑布图
        shap.plots.waterfall(shap.Explanation(
            values=shap_values_single,  # 确保是 (n_features,) 而不是 (n_features, n_classes)
            base_values=expected_value,
            data=original_feature_values.iloc[sample_idx],
            feature_names=original_feature_values.columns.tolist()
        ))
        plt.tight_layout()
        st.pyplot(plt.gcf())  # 直接显示，不需要保存图片

