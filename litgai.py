import streamlit as st
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, RegularGridInterpolator
import io
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
from scipy.interpolate import griddata

# 设置页面标题和图标
st.set_page_config(
    page_title="Digital Hydrogen-P",
    page_icon=r"D:\pythonProject3\streamlit\f8523a5d627f3875452fa1ece3b4d30.png",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown(
    """
    <style>
        div.stButton > button {
            width: 100%; 
            height: 40px; 
            font-size: 25px;
            background-color: #f0f2f6;  
            font-weight: bold;  
            border-radius: 20px;  
            border: none;
        }
        div.stButton > button:hover {
            background-color: #e3e5e7;
            opacity: 0.85;
        }   
        div.stButton > button:focus {
            background-color: #e3e5e7;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 初始化 session_state
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

if "show_sub_buttons" not in st.session_state:
    st.session_state.show_sub_buttons = False

if "df_results" not in st.session_state:
    st.session_state.df_results = None

if "show_plot" not in st.session_state:
    st.session_state.show_plot = False


# 侧边栏处理
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


logo_path = r"D:\pythonProject3\streamlit\f8523a5d627f3875452fa1ece3b4d30.png"
your_base64_logo = image_to_base64(logo_path)

st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{your_base64_logo}" width="250"/>
        <h1 style="font-size: 27px; font-weight: bold; margin-top: 10px;">Digital Hydrogen-P</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# 侧边栏导航
if st.sidebar.button("🏠 Home"):
    st.session_state.page = "🏠 Home"
    st.session_state.show_sub_buttons = False

if st.sidebar.button("⚙️ 功能"):
    st.session_state.page = "⚙️ 功能"
    st.session_state.show_sub_buttons = not st.session_state.show_sub_buttons

if st.session_state.show_sub_buttons:
    if st.sidebar.button("📌 定值查询"):
        st.session_state.page = "📌 定值查询"
    if st.sidebar.button("📏 范围查询"):
        st.session_state.page = "📏 范围查询"
    if st.sidebar.button("🔬 实验数据查询"):
        st.session_state.page = "🔬 实验数据查询"


# 数据加载函数
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # 清理所有列名空格
    return df


# 插值函数
def interpolate_property(pressure, temperature, data_df, property_name, method='griddata'):
    # 确保必要列存在
    required_cols = ['pressure', 'temperature', property_name]
    if not all(col in data_df.columns for col in required_cols):
        return "DataFrame missing required columns."

    for col in required_cols:
        data_df[col] = pd.to_numeric(data_df[col], errors='coerce')

    # 移除包含 NaN 的行
    data_df.dropna(subset=required_cols, inplace=True)

    # 获取网格数据
    points = data_df[['pressure', 'temperature']].values
    values = data_df[property_name].values

    try:
        if method == 'griddata':  # 替代 interp2d
            result = griddata(points, values, (pressure, temperature), method='linear')
            return result if result is not None else "无法插值"
        elif method == 'RegularGridInterpolator':  # 原有方法
            pressure_vals = np.sort(data_df['pressure'].unique())
            temperature_vals = np.sort(data_df['temperature'].unique())
            property_matrix = data_df.pivot_table(index='temperature', columns='pressure', values=property_name).values
            interp_func = RegularGridInterpolator(
                (temperature_vals, pressure_vals),
                property_matrix,
                method='linear',
                bounds_error=False
            )
            return interp_func([[temperature, pressure]])[0]
        elif method == 'nearest':  # 另一种新的插值方式
            result = griddata(points, values, (pressure, temperature), method='nearest')
            return result if result is not None else "无法插值"
        else:
            return "Unsupported interpolation method."
    except Exception as e:
        return f"Interpolation failed with error: {str(e)}"


# 范围查询函数
def generate_table(min_pressure, max_pressure, step_size, min_temperature, max_temperature, method='interp2d'):
    # 将步长转换为浮点数以确保正确处理
    step_size = float(step_size)

    # 处理浮点数情况
    # 生成压力范围
    num_p = int(round((max_pressure - min_pressure) / step_size)) + 1
    pressures = []
    for p in np.linspace(min_pressure, max_pressure, num_p):
        if float(p).is_integer():
            pressures.append(int(p))
        else:
            pressures.append(round(p, 2))

    # 生成温度范围
    num_t = int(round((max_temperature - min_temperature) / step_size)) + 1
    temperatures = []
    for t in np.linspace(min_temperature, max_temperature, num_t):
        if float(t).is_integer():
            temperatures.append(int(t))
        else:
            temperatures.append(round(t, 2))

    thermal_df = load_data(r'D:\pythonProject3\streamlit\thermal_conductivity.csv')
    viscosity_df = load_data(r'D:\pythonProject3\streamlit\viscosity.csv')
    diffusion_df = load_data(r'D:\pythonProject3\streamlit\kuosanxishu.csv')

    table_data = []
    for pressure in pressures:
        for temperature in temperatures:
            row = {'Pressure': pressure, 'Temperature': temperature}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                # 热导率
                thermal_val = interpolate_property(pressure, temperature, thermal_df, prop, method)
                if isinstance(thermal_val, str):
                    st.error(f"热导率插值错误({pressure}MPa, {temperature}K, {prop}): {thermal_val}")
                    continue
                row[f'Thermal Conductivity {prop}'] = round(float(thermal_val), 3)

                # 粘度
                viscosity_val = interpolate_property(pressure, temperature, viscosity_df, prop, method)
                if isinstance(viscosity_val, str):
                    st.error(f"粘度插值错误({pressure}MPa, {temperature}K, {prop}): {viscosity_val}")
                    continue
                row[f'Viscosity {prop}'] = round(float(viscosity_val), 3)

                # 扩散系数
                diffusion_val = interpolate_property(pressure, temperature, diffusion_df, prop, method)
                if isinstance(diffusion_val, str):
                    st.error(f"扩散系数插值错误({pressure}MPa, {temperature}K, {prop}): {diffusion_val}")
                    continue
                row[f'Diffusion {prop}'] = round(float(diffusion_val), 9)

            table_data.append(row)
    return pd.DataFrame(table_data)


# 页面逻辑
if st.session_state.page == "🏠 Home":
    st.image(r"D:\pythonProject3\streamlit\8c3351f1e7b958ef4fdc8dfb9d5d99f.png", width=400)
    st.title("Digital Hydrogen-P")
    st.write("""
        **欢迎来到 Digital Hydrogen-P**  
        
        本网站致力于提供便捷、准确的氢气热物性数据查询服务。您可以快速进行数据查询、插值计算和可视化分析，深入了解氢气的热导率、粘度及扩散系数热物性信息。适用于科研人员、工程师及学生，帮助您更高效地完成氢气热物性相关的研究与工程设计。
    """)
    st.write("""
        **功能介绍**：
        - **定值查询**：轻松快速地输入指定压强和温度，精准获取氢气的热导率、粘度及扩散系数数值
        - **范围查询**：支持自定义压力和温度范围及步长，批量获取氢气的热物性数据，并通过表格与图形直观展示结果，满足多种分析需求。
        - **模型预测**：提供权威的实验数据来源，允许用户按文献标题选择和浏览热导率、粘度、扩散系数的实验数据，并支持便捷的数据导出功能。
        - **模型预测**：内置交互式数据可视化工具，支持数据二维、三维可视化展示，帮助用户直观理解数据分布与趋势。
    """)
    # st.write("""
    #     **示例文件下载**：
    #     [GitHub 参考资料](https://github.com/withand123/HydrogenCell-Life)
    # """)

elif st.session_state.page == "📏 范围查询":
    st.title("📏 范围查询")
    col1, col2 = st.columns(2)
    with col1:
        min_pressure = st.number_input("最小压强 (MPa)", min_value=0.0, step=5.0, format="%.1f", value=None)
        max_pressure = st.number_input("最大压强 (MPa)", min_value=min_pressure if min_pressure else 0.0, step=5.0, format="%.1f", value=None)
    with col2:
        min_temperature = st.number_input("最小温度 (K)", min_value=0.0, step=5.0, format="%.1f", value=None)
        max_temperature = st.number_input("最大温度 (K)", min_value=min_temperature if min_temperature else 0.0, step=5.0, format="%.1f", value=None)

    step_size = st.selectbox("步长", [1, 2, 5, 10, 20, 50, 100])
    interpolation_method = st.selectbox("插值方法", ["griddata", "RegularGridInterpolator", "nearest"])


    # 触发查询功能
    if st.button("🔍 查询"):
        if min_pressure is None or max_pressure is None or min_temperature is None or max_temperature is None:
            st.warning("请输入完整的压强和温度范围")
        elif min_pressure > max_pressure or min_temperature > max_temperature:
            st.warning("最小压强不能大于最大压强，最小温度不能大于最大温度")
        else:
            st.session_state.df_results = generate_table(
                min_pressure, max_pressure, step_size, min_temperature, max_temperature, interpolation_method)
            if not st.session_state.df_results.empty:
                st.success("查询成功！")
            else:
                st.error("未生成有效查询结果，请检查输入参数或数据文件。")

    # 确保有查询结果后再显示功能
    if st.session_state.df_results is not None and not st.session_state.df_results.empty:
        # 创建格式化副本
        formatted_df = st.session_state.df_results.copy()


        # 识别需要格式化的列
        diffusion_cols = [col for col in formatted_df.columns if 'Diffusion' in col]

        # 应用科学计数法格式化
        for col in diffusion_cols:
            formatted_df[col] = formatted_df[col].apply(
                lambda x: "{:.3e}".format(x) if isinstance(x, (int, float)) else x)

        # 显示带格式的表格
        st.dataframe(formatted_df.style.set_properties(
            subset=diffusion_cols, ** {'text-align': 'center', 'font-family': 'monospace'}
        ))

        # 按钮在同一行
        col1, col2, col3 = st.columns(3)

        with col1:
            # 生成 Excel 文件
            excel_data = io.BytesIO()
            with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
                st.session_state.df_results.to_excel(writer, index=False)
            excel_data.seek(0)  # 重要：重置指针位置
            st.download_button(
                label="📥 下载 Excel",
                data=excel_data,
                file_name="定值查询结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            # 生成 TXT 文件
            txt_data = st.session_state.df_results.to_csv(sep='\t', index=False).encode('utf-8')
            st.download_button(
                label="📥 下载 TXT",
                data=txt_data,
                file_name="定值查询结果.txt",
                mime="text/plain"
            )

        with col3:
            if st.button("📊 绘图"):
                st.session_state.show_plot = True

        # 结果可视化
        if st.session_state.show_plot:
            st.subheader("📊 结果可视化")

            plt.rcParams.update({
                'font.size': 16,  # 全局字体大小
                'axes.titlesize': 16,  # 子图标题大小
                'axes.labelsize': 15,  # 坐标轴标签大小
                'xtick.labelsize': 16,  # X轴刻度
                'ytick.labelsize': 16,  # Y轴刻度
                'legend.fontsize': 14  # 图例
            })

            # 让用户调整点的大小
            marker_size = st.slider("选择点的大小", min_value=2, max_value=10, value=5)

            # **增加图的大小**
            fig, axes = plt.subplots(3, 1, figsize=(12, 16))  # 改为3行1列

            if min_temperature == max_temperature:
                x_axis = 'Pressure'
                x_label = "Pressure (MPa)"
            elif min_pressure == max_pressure:
                x_axis = 'Temperature'
                x_label = "Temperature (K)"
            else:
                x_axis = None

            if x_axis:
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[0].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Thermal Conductivity {prop}'],
                        marker='o', markersize=marker_size, linestyle='-', label=f'Thermal {prop}'
                    )
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[1].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Viscosity {prop}'],
                        marker='s', markersize=marker_size, linestyle='--', label=f'Viscosity {prop}'
                    )
                for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                    axes[2].plot(
                        st.session_state.df_results[x_axis],
                        st.session_state.df_results[f'Diffusion {prop}'],
                        marker='^', markersize=marker_size, linestyle=':', label=f'Diffusion {prop}'
                    )

                axes[0].set_xlabel(x_label)
                axes[0].set_ylabel("Thermal Conductivity (W/m·K)")
                axes[0].legend(loc='upper left')  # **图例固定在左上角**
                axes[0].set_title("热导率")

                axes[1].set_xlabel(x_label)
                axes[1].set_ylabel("Viscosity (μPa·s)")
                axes[1].legend(loc='upper left')  # **图例固定在左上角**
                axes[1].set_title("粘度")

                axes[2].set_xlabel(x_label)
                axes[2].set_ylabel("扩散系数 (m$^{2}$/s)")
                axes[2].legend(loc='upper left')  # **图例固定在左上角**
                axes[2].set_title("扩散系数")
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
                plt.tight_layout()
                st.pyplot(fig)


elif st.session_state.page == "📌 定值查询":
    st.title("📌 定值查询")

    # 新布局结构
    st.subheader("🔢 输入参数")

    # 第一行：压力温度输入框并排
    col_pres_temp = st.columns(2)
    with col_pres_temp[0]:
        pressure = st.number_input("输入压强 (MPa)", min_value=0.0, step=5.0, format="%.2f", value=None)
    with col_pres_temp[1]:
        temperature = st.number_input("输入温度 (K)", min_value=0.0, step=5.0, format="%.1f", value=None)

    # 第二行：插值方法和查询按钮并排
    col_method_btn = st.columns([2, 2])
    with col_method_btn[0]:
        interpolation_method = st.selectbox("选择插值方法",
                                            ["griddata", "RegularGridInterpolator", "nearest"],
                                            key="method_selectbox")
    with col_method_btn[1]:
        st.write("")  # 垂直对齐
        st.write("")
        query_clicked = st.button("🔍 立即查询", use_container_width=True)

    # 触发查询按钮
    if query_clicked:
        if pressure is None or temperature is None:
            st.warning("请输入压强和温度")

        else:
            thermal_df = load_data(r'D:\pythonProject3\streamlit\thermal_conductivity.csv')
            viscosity_df = load_data(r'D:\pythonProject3\streamlit\viscosity.csv')
            diffusion_df = load_data(r'D:\pythonProject3\streamlit\kuosanxishu.csv')
            error_occurred = False

            thermal_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, thermal_df, prop, interpolation_method)
                if isinstance(result, str):
                    st.error(f"热导率 {prop} 计算错误: {result}")
                    error_occurred = True
                thermal_results[prop] = result

            viscosity_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, viscosity_df, prop, interpolation_method)
                if isinstance(result, str):
                    st.error(f"粘度 {prop} 计算错误: {result}")
                    error_occurred = True
                viscosity_results[prop] = result

            diffusion_results = {}
            for prop in ['ML', 'Buch', 'Cracknell', 'Vrabec', 'Tahery', 'NIST']:
                result = interpolate_property(pressure, temperature, diffusion_df, prop, interpolation_method)
                if isinstance(result, str):
                    st.error(f"扩散系数 {prop} 计算错误: {result}")
                    error_occurred = True
                diffusion_results[prop] = result

            if not error_occurred:
                st.session_state.thermal_results = thermal_results
                st.session_state.viscosity_results = viscosity_results
                st.session_state.diffusion_results = diffusion_results  # 存储扩散结果
                st.session_state.show_results = True

    if st.session_state.get("show_results", False):
        st.subheader("📊 计算结果")

        # 三列布局展示结果
        col_thermal, col_visc, col_diff = st.columns(3)

        with col_thermal:
            st.markdown("<h4 style='font-size:16px; margin-bottom:12px;'>热导率 (W/m·K)</h4>", unsafe_allow_html=True)
            for name, val in st.session_state.thermal_results.items():
                st.write(f"**{name}**: {val:.3f}")

        with col_visc:
            st.markdown("<h4 style='font-size:16px; margin-bottom:12px;'>粘度 (μPa·s)</h4>", unsafe_allow_html=True)
            for name, val in st.session_state.viscosity_results.items():
                st.write(f"**{name}**: {val:.3f}")

        with col_diff:
            st.markdown("<h4 style='font-size:16px; margin-bottom:12px;'>扩散系数 (m²/s)</h4>", unsafe_allow_html=True)
            for name, val in st.session_state.diffusion_results.items():
                st.write(f"**{name}**: {val:.3e}")

elif st.session_state.page == "⚙️ 功能":
    st.title("⚙️ 功能")
    st.write("请选择侧边栏的子功能按钮进行操作")


elif st.session_state.page == "🔬 实验数据查询":
    st.title("🔬 实验数据查询")

    # 加载实验数据
    thermal_df = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanredaol.csv')
    viscosity_df = pd.read_csv(r'D:\pythonProject3\streamlit\shiyanniandu.csv')

    # 获取所有文章标题
    thermal_article_titles = thermal_df['redaoarticle title'].unique()
    viscosity_article_titles = viscosity_df['nianduarticle title'].unique()

    # 选择热导率文章
    selected_thermal_article = st.selectbox("选择热导率文章", [""] + list(thermal_article_titles))

    # 选择粘度文章
    selected_viscosity_article = st.selectbox("选择粘度文章", [""] + list(viscosity_article_titles))
    if st.button("📊 绘制全部热导率实验数据的 3D 图"):
        fig = go.Figure(data=[go.Scatter3d(
            x=thermal_df['pressure'],
            y=thermal_df['temperature'],
            z=thermal_df['redaoexperimentalvalue'],
            mode='markers',
            marker=dict(size=5, color=thermal_df['redaoexperimentalvalue'], colorscale='Viridis')
        )])

        fig.update_layout(
            title="热导率实验数据 3D 可视化",
            scene=dict(
                xaxis_title="压力 (MPa)",
                yaxis_title="温度 (K)",
                zaxis_title="热导率实验值（W/(m·K)）"
            )
        )

        st.plotly_chart(fig)

    if st.button("📊 绘制全部粘度实验数据的 3D 图"):
        fig = go.Figure(data=[go.Scatter3d(
            x=viscosity_df['pressure'],
            y=viscosity_df['temperature'],
            z=viscosity_df['nianduexperimentalvalue'],
            mode='markers',
            marker=dict(size=5, color=viscosity_df['nianduexperimentalvalue'], colorscale='Viridis')
        )])

        fig.update_layout(
            title="粘度实验数据 3D 可视化",
            scene=dict(
                xaxis_title="压力 (MPa)",
                yaxis_title="温度 (K)",
                zaxis_title="粘度实验值（μPa·s）"
            )
        )

        st.plotly_chart(fig)
    table_data = None
    if selected_thermal_article:
        table_data = thermal_df[thermal_df['redaoarticle title'] == selected_thermal_article][
            ['pressure', 'temperature', 'redaoexperimentalvalue']]
        table_data.columns = ['Pressure', 'Temperature', 'Experimental Value']

    if selected_viscosity_article:
        table_data = viscosity_df[viscosity_df['nianduarticle title'] == selected_viscosity_article][
            ['pressure', 'temperature', 'nianduexperimentalvalue']]
        table_data.columns = ['Pressure', 'Temperature', 'Experimental Value']

    if table_data is not None and not table_data.empty:
        # 使用 st.data_editor() 代替 st.dataframe()
        st.data_editor(
            table_data.style.set_properties(**{'text-align': 'center'}),  # 设置文本居中
            height=400,  # 限制高度，启用滚动
            use_container_width=True  # 让表格自适应宽度
        )

        # **📥 下载功能**
        col1, col2 = st.columns(2)

        with col1:
            excel_data = io.BytesIO()
            with pd.ExcelWriter(excel_data, engine='xlsxwriter') as writer:
                table_data.to_excel(writer, index=False)
            excel_data.seek(0)
            st.download_button("📥 下载 Excel", data=excel_data, file_name="实验数据.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col2:
            txt_data = table_data.to_csv(sep='\t', index=False).encode('utf-8')
            st.download_button("📥 下载 TXT", data=txt_data, file_name="实验数据.txt", mime="text/plain")

        # **📊 绘制三维图**
        if st.button("📊 绘制 3D 图"):
            fig = go.Figure(data=[go.Scatter3d(
                x=table_data['Pressure'],
                y=table_data['Temperature'],
                z=table_data['Experimental Value'],
                mode='markers',
                marker=dict(size=5, color=table_data['Experimental Value'], colorscale='Viridis')
            )])

            fig.update_layout(
                title="实验数据 3D 可视化",
                scene=dict(
                    xaxis_title="压力 (MPa)",
                    yaxis_title="温度 (K)",
                    zaxis_title="实验值"
                )
            )

            st.plotly_chart(fig)