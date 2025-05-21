import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Стили кнопок (меню)
st.markdown("""
<style>
    button {
        background: none !important;
        border: none !important;
        box-shadow: none !important;
    }
    button:hover {
        color: #b11226 !important;
        transform: scale(1.02)
    }
    button:active {
        transform: scale(0.98);
    }
    button:focus {
        color: #b11226 !important;
    } 
</style>
""", unsafe_allow_html=True)

# График сравнения accuracy
sns.set_theme()
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Сравнение accuracy", fontsize=20, pad=14, color='#555555')
sns.barplot(
    x=['KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'MLPClassifier'],
    y=[0.250, 0.240, 0.250, 0.290, 0.390],
    ax=ax,
    color='#b11226',
    width=0.4
)
ax.set_xticklabels(
    ['KNeighborsClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier', 'SVC', 'MLPClassifier'],
    rotation=20,
    ha='right',
    fontsize=14,
    color='gray'
)
ax.set_ylabel('Accuracy', fontsize=14, color='gray')
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
ax.grid(False, axis='x')
ax.grid(True, axis='y', color='lightgrey')


def set_page(page):
    # Выбор страницы
    st.session_state.page = page


# Левое меню
with st.sidebar:
    st.sidebar.title("Модели обучения")
    st.button("Сравнение моделей", on_click=set_page, args=('home',))
    st.button("KNeighborsClassifier", on_click=set_page, args=('KNeighborsClassifier',))
    st.button("RandomForestClassifier", on_click=set_page, args=('RandomForestClassifier',))
    st.button("GradientBoostingClassifier", on_click=set_page, args=('GradientBoostingClassifier',))
    st.button("SVC", on_click=set_page, args=('SVC',))
    st.button("MLPClassifier", on_click=set_page, args=('MLPClassifier',))

# Стартовая страница
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Заполнение страниц
if st.session_state.page == 'home':
    st.title("Многоклассовая классификация")
    st.write("Реализация многоклассовой классификации рукописных цифр из датасета "
             "Street View House Numbers (SVHN, http://ufldl.stanford.edu/housenumbers/) "
             "с помощью библиотеки sklearn.\n\n"
             "Информация по каждой модели доступна по ссылкам в левом меню.")
    st.header("Сравнение моделей")
    st.write("На графике представлены метрики accuracy обученных моделей.")
    st.pyplot(fig)
    st.write("Наилучший результат показала модель MLPClassifier.")

elif st.session_state.page == 'KNeighborsClassifier':
    st.header("KNeighborsClassifier")
    st.markdown("**Лучшие гиперпараметры:** {'n_neighbors': 5, 'weights': 'distance'}")
    st.markdown("**Лучшая accuracy на обучающих данных:** 0.260")
    st.markdown("<span style='color: #b11226'>**Accuracy на тестовых данных:**</span> 0.250", unsafe_allow_html=True)
    st.code("from sklearn.neighbors import KNeighborsClassifier\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "model = KNeighborsClassifier()\n\n"
            "param_grid = {\n"
            "   'n_neighbors': [3, 5, 10],\n"
            "   'weights': ['uniform', 'distance'],\n"
            "}\n\n"
            "clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')\n"
            "clf.fit(train_X, train_y)\n"
            "prediction = clf.predict(test_X)\n"
            "accuracy = accuracy_score(test_y, prediction)")

elif st.session_state.page == 'RandomForestClassifier':
    st.header("RandomForestClassifier")
    st.markdown("**Лучшие гиперпараметры:** {'max_depth': None, 'max_features': 0.8}")
    st.markdown("**Лучшая accuracy на обучающих данных:** 0.346")
    st.markdown("<span style='color: #b11226'>**Accuracy на тестовых данных:**</span> 0.240", unsafe_allow_html=True)
    st.code("from sklearn.ensemble import RandomForestClassifier\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "model = RandomForestClassifier()\n\n"
            "param_grid = {\n"
            "   'max_depth': [5, 10, None],\n"
            "   'max_features': ['sqrt', 0.8],\n"
            "}\n\n"
            "clf = GridSearchCV(model, param_grid, scoring='accuracy')\n"
            "clf.fit(train_X, train_y)\n"
            "prediction = clf.predict(test_X)\n"
            "accuracy = accuracy_score(test_y, prediction)")

elif st.session_state.page == 'GradientBoostingClassifier':
    st.header("GradientBoostingClassifier")
    st.markdown("**Лучшие гиперпараметры:** {'max_depth': 8, 'max_features': 'sqrt'}")
    st.markdown("**Лучшая accuracy на обучающих данных:** 0.284")
    st.markdown("<span style='color: #b11226'>**Accuracy на тестовых данных:**</span> 0.250", unsafe_allow_html=True)
    st.code("from sklearn.ensemble import GradientBoostingClassifier\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.metrics import accuracy_score\n\n"
            "model = GradientBoostingClassifier(n_iter_no_change=10)\n\n"
            "param_grid = {\n"
            "   'max_depth': [3, 5, 8],\n"
            "   'max_features': ['sqrt', 0.8],\n"
            "}\n\n"
            "clf = GridSearchCV(model, param_grid, scoring='accuracy')\n"
            "clf.fit(train_X, train_y)\n"
            "prediction = clf.predict(test_X)\n"
            "accuracy = accuracy_score(test_y, prediction)")

elif st.session_state.page == 'SVC':
    st.header("SVC")
    st.markdown("**Лучшие гиперпараметры:** {'C': 10, 'kernel': 'rbf'}")
    st.markdown("**Лучшая accuracy на обучающих данных:** 0.318")
    st.markdown("<span style='color: #b11226'>**Accuracy на тестовых данных:**</span> 0.290", unsafe_allow_html=True)
    st.code("from sklearn.svm import SVC\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.metrics import accuracy_score\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "scaler = StandardScaler()\n"
            "train_X_scaled = scaler.fit_transform(train_X)\n"
            "test_X_scaled = scaler.transform(test_X)\n\n"
            "model = SVC()\n\n"
            "param_grid = {\n"
            "   'C': [0.1, 1, 10],\n"
            "   'kernel': ['rbf', 'poly'],\n"
            "}\n\n"
            "clf = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')\n"
            "clf.fit(train_X_scaled, train_y)\n"
            "prediction = clf.predict(test_X_scaled)\n"
            "accuracy = accuracy_score(test_y, prediction)")

elif st.session_state.page == 'MLPClassifier':
    st.header("MLPClassifier")
    st.markdown("**Лучшие гиперпараметры:** {'alpha': 0.001, 'hidden_layer_sizes': (128, 64)}")
    st.markdown("**Лучшая accuracy на обучающих данных:** 0.373")
    st.markdown("<span style='color: #b11226'>**Accuracy на тестовых данных:**</span> 0.390", unsafe_allow_html=True)
    st.code("from sklearn.neural_network import MLPClassifier\n"
            "from sklearn.model_selection import GridSearchCV\n"
            "from sklearn.metrics import accuracy_score\n"
            "from sklearn.preprocessing import StandardScaler\n\n"
            "scaler = MLPClassifier(max_iter=500, early_stopping=True)\n"
            "train_X_scaled = scaler.fit_transform(train_X)\n"
            "test_X_scaled = scaler.transform(test_X)\n\n"
            "model = SVC()\n\n"
            "param_grid = {\n"
            "   'hidden_layer_sizes': [(100,), (128, 64)]\n"
            "   'alpha': [0.0001, 0.001],\n"
            "}\n\n"
            "clf = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')\n"
            "clf.fit(train_X_scaled, train_y)\n"
            "prediction = clf.predict(test_X_scaled)\n"
            "accuracy = accuracy_score(test_y, prediction)")
