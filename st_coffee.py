import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 모델 로드
model = joblib.load('./model/optimized_rf_model_coffee.pkl')
classifier = joblib.load('./model/customer_segment_classifier.pkl')
segment_profiles = joblib.load('./model/customer_segment_profiles.pkl')
category_encoder = joblib.load('./model/category_encoder.pkl')

# 페이지 설정
st.set_page_config(page_title='커피숍 일일 매출 예측기', page_icon=':coffee:', layout='wide')

# 사이드바 메뉴
menu = st.sidebar.selectbox("메뉴 선택", ["매출 예측", "전략 수립"])

# ------------------------------
# 매출 예측 탭
# ------------------------------
if menu == '매출 예측':
    st.title('☕ 커피숍 일일 매출 예측기')
    st.write("입력값을 바탕으로 오늘의 예상 매출을 예측합니다.")

    number_customers = st.number_input("일일 방문객 수(명)", min_value=50, value=100, max_value=500, step=1)
    avg_order_value = st.number_input("1인당 평균 주문 금액($)", min_value=2.5, value=10.0, max_value=10.0, step=0.5)
    operating_hours = st.number_input("운영 시간(시간)", min_value=6.0, value=8.0, max_value=18.0, step=0.5)
    number_employees = st.number_input("종업원 수(명)", min_value=2, value=5, max_value=15, step=1)
    marketing_spend = st.number_input("일일 마케팅 비용($)", min_value=10.0, value=50.0, max_value=500.0, step=1.0)
    foot_traffic = st.number_input("시간당 유동인구(명)", min_value=50, value=500, max_value=1000, step=1)

    if st.button("일일 매출 예측"):
        input_data = np.array([[number_customers, avg_order_value, operating_hours,
                                number_employees, marketing_spend, foot_traffic]])
        prediction = model.predict(input_data)
        st.success(f"오늘의 예상 매출: 약 ${prediction[0]:,.2f}")

# ------------------------------
# 전략 수립 탭
# ------------------------------
elif menu == '전략 수립':
    st.title("💡 전략 수립 도우미")
    st.write("매출을 입력하면 고객 세그먼트를 분류하고 타겟 특성과 전략을 제시합니다.")

    def scale_to_100(x, original_min=305.1, original_max=4675.86):
        return (x - original_min) / (original_max - original_min) * 100

    sales = st.number_input("예상 매출액을 입력하세요", min_value=305.1, max_value=4675.86, step=10.0, value=1000.0)
    
    score = scale_to_100(sales)

    if st.button("전략 추천 받기"):
        # 입력된 소비 점수만을 사용
        sample = pd.DataFrame([{'spending_score': score}])
        pred_segment = classifier.predict(sample)[0]

        st.subheader(f"🎯 예측된 고객 세그먼트: **{pred_segment}**")

        profile = segment_profiles[segment_profiles['Segment'] == pred_segment].iloc[0]
        gender_str = profile['Gender']
        category_str = category_encoder.inverse_transform([int(profile['Preferred Category'])])[0]


        st.markdown(f"""
        - **평균 나이**: {profile['age']:.1f}세  
        - **월 소득**: ${profile['income']:.2f}k  
        - **평균 소비 점수**: {profile['spending_score']:.1f}  
        - **멤버십 가입 기간**: {profile['membership_years']:.1f}년  
        - **주요 성별**: {gender_str}  
        - **선호 카테고리**: {category_str}
        """)

        st.subheader("📌 추천 마케팅 전략")
        if pred_segment == '가격 민감형 소비집단':
            st.markdown("- **할인 메뉴** 및 **SNS 이벤트**로 관심 유도")
        elif pred_segment == '일반 소비집단':
            st.markdown("- **포인트 적립** 및 **재방문 쿠폰** 제공")
        elif pred_segment == '고소비집단':
            st.markdown("- **프리미엄 상품 구성** 및 **고급 인테리어** 강조")
        else:
            st.markdown("- **VIP 라운지** 및 **맞춤형 혜택** 제공")
