import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

def load_data():
    data = pd.read_csv('Shopping Trends And Customer Behaviour Dataset.csv')
    data.drop('Unnamed: 0', axis=1, inplace=True)
    return data

import plotly.graph_objects as go  # Required for custom Plotly figures

def show_general_plots(data, pie_btn, hist_btn, scatter_btn, bar_btn,
                           barwithlegend_btn, lollipopchart_btn,
                           stackedbar_btn, bubblechart_btn):
    st.header("General Plots")

    if pie_btn:
        fig = px.pie(data, names='Category', title='Category Distribution')
        st.plotly_chart(fig, use_container_width=True)

    if hist_btn:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(data['Review Rating'],
                bins=10, color='pink',
                edgecolor='black')
        ax.set_xlabel('Review Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Customer Review Rating Trends')
        ax.bar_label(plt.gca().containers[0])
        st.pyplot(fig)

    if scatter_btn:
        item_counts = data['Item Purchased'].value_counts()
        fig = px.scatter(x=item_counts.index,
                      y=item_counts.values,
                      labels={'x': 'Item', 'y': 'Count'},)
        st.plotly_chart(fig)

    if bar_btn:
        st.subheader("Category Distribution by Season")
        season_clothing_counts = data.groupby('Season')['Category'].value_counts().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(10, 4))
        season_clothing_counts.plot(kind='bar', stacked=False, ax=ax)
        ax.set_xlabel('Season')
        ax.set_ylabel('Number of Categories')
        ax.set_title('Categories Distribution in Each Season')
        ax.legend(title='Categories', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    if barwithlegend_btn:
        fig = go.Figure()
        plt.subplots(figsize=(10, 2))
        for status in data['Subscription Status'].unique():
            shipping_counts = data[data['Subscription Status'] == status]['Shipping Type'].value_counts()
            fig.add_trace(go.Bar(
                x=shipping_counts.index,
                y=shipping_counts.values,
                name=status
            ))
        fig.update_layout(
            title='Subscription Status in Each Shipping Type',
            xaxis_title='Shipping Type',
            yaxis_title='Count',
            barmode='group',
            legend_title='Subscription Status',
            width=800,
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig)

    if lollipopchart_btn:
        frequency_counts = data['Frequency of Purchases'].value_counts().reset_index()
        frequency_counts.columns = ['Frequency of Purchases', 'Count']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=frequency_counts['Frequency of Purchases'],
            y=frequency_counts['Count'],
            mode='markers',
            marker=dict(size=10),
            name='Frequency'
        ))
        for index, row in frequency_counts.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['Frequency of Purchases'], row['Frequency of Purchases']],
                y=[0, row['Count']],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
        fig.update_layout(
            title='Frequency of Purchases vs. Count',
            xaxis_title='Frequency of Purchases',
            yaxis_title='Count',
            showlegend=True
        )
        st.plotly_chart(fig)

    if stackedbar_btn:
        discount_prev_purchases = data.groupby(['Previous Purchases', 'Discount Applied']).size().unstack(fill_value=0)
        fig = go.Figure(data=[
            go.Bar(name=col, x=discount_prev_purchases.index, y=discount_prev_purchases[col])
            for col in discount_prev_purchases.columns
        ])
        fig.update_layout(
            barmode='stack',
            title='Discount Applied vs. Previous Purchases',
            xaxis_title='Previous Purchases',
            yaxis_title='Count',
            legend_title='Discount Applied'
        )
        st.plotly_chart(fig)

    if  bubblechart_btn:
        promo_category_counts = data.groupby(['Promo Code Used', 'Category']).size().reset_index(name='Count')
        fig = px.scatter(promo_category_counts,
                         x='Promo Code Used',
                         y='Count',
                         color='Category',
                         size='Count',
                         title='Relationship between Promo Code Used and Category')
        st.plotly_chart(fig)


def show_feature_scaling(data, minmax_btn, standard_btn):
    st.header("Feature Scaling")
    cols = ['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']

    if minmax_btn:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data[cols])
        scaled_df = pd.DataFrame(scaled, columns=[f"{c} (Scaled)" for c in cols])
        st.dataframe(scaled_df)

    if standard_btn:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(data[cols])
        scaled_df = pd.DataFrame(scaled, columns=[f"{c} (Z-Scaled)" for c in cols])
        st.dataframe(scaled_df)

def show_regression(data, category_filter, location_filter, season_filter, linreg_btn, polyreg_btn):
    st.header("Regression Analysis")

    filtered_df = data.copy()
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["Category"] == category_filter]
    if location_filter != "All":
        filtered_df = filtered_df[filtered_df["Location"] == location_filter]
    if season_filter != "All":
        filtered_df = filtered_df[filtered_df["Season"] == season_filter]

    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        return

    X = filtered_df[["Age"]]
    y = filtered_df["Review Rating"]
    fig = px.scatter(filtered_df, x="Age", y="Review Rating", color="Gender", title="Age vs Review Rating")

    if linreg_btn:
        lr = LinearRegression().fit(X, y)
        fig.add_scatter(x=X["Age"], y=lr.predict(X), mode='lines', name="Linear Regression", line=dict(color='red'))

    if polyreg_btn:
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        pr = LinearRegression().fit(X_poly, y)
        pred = pr.predict(X_poly)
        sorted_idx = X["Age"].argsort()
        fig.add_scatter(x=X["Age"].iloc[sorted_idx], y=pred[sorted_idx], mode='lines', name="Polynomial", line=dict(color='green'))

    st.plotly_chart(fig, use_container_width=True)

def show_classifier(df, classifier_name, logistic_C, logistic_iter, rf_trees, rf_depth, train_btn):
    st.header("Classifier")

    # Prepare X, y
    y = df['Discount Applied']

    # Drop 'Discount Applied' from X
    X = df.drop(columns=['Discount Applied'])

    # Handle categorical variables with get_dummies
    X = pd.get_dummies(X, drop_first=True)

    # Drop rows with any missing values (if any remain)
    X = X.fillna(0)

    # Align y and X indices after fillna (just in case)
    y = y.loc[X.index]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    if classifier_name == "Logistic Regression":
        model = LogisticRegression(C=logistic_C, max_iter=logistic_iter)
    else:
        model = RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, random_state=42)

    if train_btn:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", model.score(x_test, y_test))
        st.write("Precision:", precision_score(y_test, y_pred, average='macro'))
        st.write("Recall:", recall_score(y_test, y_pred, average='macro'))

def main():
    st.set_page_config(layout="wide")
    st.title("🛍️ Shopping Trends and Customer Behaviour Dashboard")

    data = load_data()

    st.sidebar.title("📊 Navigation")
    section = st.sidebar.radio("Go to", ["Raw Dataset", "General Plots", "Feature Scaling", "Regression", "Classifier"])

    if section == "Raw Dataset":
        st.subheader("Dataset Preview")
        st.dataframe(data)

    elif section == "General Plots":
        st.sidebar.header("General Plots Controls")
        pie_btn = st.sidebar.button("Show Category Pie Chart", key="pie")
        hist_btn = st.sidebar.button("Review Rating Histogram", key="hist")
        scatter_btn = st.sidebar.button("Item Purchased scatter Plot", key="scatter")
        bar_btn = st.sidebar.button("Season vs Category Bar Plot", key="bar")
        barwithlegend_btn = st.sidebar.button("Subscription Status vs Shipping Type", key="barlegend")
        lollipopchart_btn = st.sidebar.button("Lollipop Chart: Frequency of Purchases", key="lollipop")
        stackedbar_btn = st.sidebar.button("Discount Applied vs Previous Purchases (Stacked Bar)", key="stacked")
        bubblechart_btn = st.sidebar.button("Promo Code Used vs Category (Bubble Chart)", key="bubble")

        show_general_plots(data, pie_btn, hist_btn, scatter_btn, bar_btn,
                           barwithlegend_btn, lollipopchart_btn,
                           stackedbar_btn, bubblechart_btn)
            

    elif section == "Feature Scaling":
        st.sidebar.header("Feature Scaling Controls")
        minmax_btn = st.sidebar.button("MinMax Scaling")
        standard_btn = st.sidebar.button("Standard Scaling")

        show_feature_scaling(data, minmax_btn, standard_btn)

    elif section == "Regression":
        st.sidebar.header("Regression Filters & Controls")
        category_filter = st.sidebar.selectbox("Category", ["All"] + list(data["Category"].unique()))
        location_filter = st.sidebar.selectbox("Location", ["All"] + list(data["Location"].unique()))
        season_filter = st.sidebar.selectbox("Season", ["All"] + list(data["Season"].unique()))

        linreg_btn = st.sidebar.button("Apply Linear Regression")
        polyreg_btn = st.sidebar.button("Apply Polynomial Regression")

        show_regression(data, category_filter, location_filter, season_filter, linreg_btn, polyreg_btn)

    elif section == "Classifier":
        st.sidebar.header("Classifier Controls")
        df = data.dropna().copy()
        df['Discount Applied'] = df['Discount Applied'].map({'Yes': 1, 'No': 0})

        classifier_name = st.sidebar.selectbox("Classifier", ["Logistic Regression", "Random Forest"])

        logistic_C = 1.0
        logistic_iter = 100
        rf_trees = 100
        rf_depth = 10

        if classifier_name == "Logistic Regression":
            logistic_C = st.sidebar.number_input("C", 0.01, 10.0, step=0.01, value=1.0)
            logistic_iter = st.sidebar.slider("Max Iterations", 100, 500, value=100)
        else:
            rf_trees = st.sidebar.slider("Trees", 100, 5000, step=100, value=100)
            rf_depth = st.sidebar.slider("Max Depth", 1, 20, value=10)

        train_btn = st.sidebar.button("Train Classifier")

        show_classifier(df, classifier_name, logistic_C, logistic_iter, rf_trees, rf_depth, train_btn)

if __name__ == '__main__':
    main()
