import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="期货季节性交易规律分析系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 期货品种映射（完整78个品种，使用连续合约代码）
FUTURES_MAPPING = {
    # 农产品 (大商所DCE)
    '玉米': 'C0', '大豆一号': 'A0', '大豆二号': 'B0', '豆粕': 'M0', '豆油': 'Y0',
    '棕榈油': 'P0', '鸡蛋': 'JD0', '淀粉': 'CS0', '粳米': 'RR0', '生猪': 'LH0',
    '纤维板': 'FB0', '胶合板': 'BB0', '原木': 'LG0',
    
    # 农产品 (郑商所CZCE)  
    '菜籽油': 'OI0', '菜籽': 'RS0', '菜籽粕': 'RM0', '强麦': 'WH0', '粳稻': 'JR0',
    '白糖': 'SR0', '棉花': 'CF0', '早籼稻': 'RI0', '晚籼稻': 'LR0', '苹果': 'AP0',
    '红枣': 'CJ0', '花生': 'PK0', '棉纱': 'CY0',
    
    # 工业品 (上期所SHFE)
    '螺纹钢': 'RB0', '热轧卷板': 'HC0', '线材': 'WR0', '不锈钢': 'SS0',
    '铜': 'CU0', '铝': 'AL0', '锌': 'ZN0', '铅': 'PB0', '镍': 'NI0', '锡': 'SN0',
    '黄金': 'AU0', '白银': 'AG0', '纸浆': 'SP0', '氧化铝': 'AO0', '丁二烯橡胶': 'BR0',
    
    # 工业品 (上海国际能源INE)
    '国际铜': 'BC0', '集运指数': 'EC0',
    
    # 工业品 (广期所GFEX)
    '工业硅': 'SI0', '碳酸锂': 'LC0', '多晶硅': 'PS0',
    
    # 化工品 (大商所DCE)
    '塑料': 'L0', 'PVC': 'V0', 'PP': 'PP0', '乙二醇': 'EG0', '苯乙烯': 'EB0',
    '液化石油气': 'PG0',
    
    # 化工品 (郑商所CZCE)
    'PTA': 'TA0', '甲醇': 'MA0', '玻璃': 'FG0', '尿素': 'UR0', '纯碱': 'SA0',
    '短纤': 'PF0', '烧碱': 'SH0', '对二甲苯': 'PX0', '瓶片': 'PR0',
    
    # 能源品 (上期所SHFE)
    '燃料油': 'FU0', '沥青': 'BU0', '天然橡胶': 'RU0',
    
    # 能源品 (上海国际能源INE)
    '原油': 'SC0', '低硫燃料油': 'LU0', '20号胶': 'NR0',
    
    # 黑色系 (大商所DCE)
    '铁矿石': 'I0', '焦炭': 'J0', '焦煤': 'JM0',
    
    # 黑色系 (郑商所CZCE)
    '硅铁': 'SF0', '锰硅': 'SM0',
    
    # 金融期货 (中金所CFFEX)
    '沪深300': 'IF0', '上证50': 'IH0', '中证500': 'IC0', '中证1000': 'IM0',
    '5年期国债': 'TF0', '2年期国债': 'TS0'
}

# 品种分类（完整78个品种）
CATEGORY_MAPPING = {
    '农产品': ['玉米', '大豆一号', '大豆二号', '豆粕', '豆油', '棕榈油', '鸡蛋', '淀粉', '粳米', '生猪',
              '纤维板', '胶合板', '原木', '菜籽油', '菜籽', '菜籽粕', '强麦', '粳稻', '白糖', '棉花',
              '早籼稻', '晚籼稻', '苹果', '红枣', '花生', '棉纱'],
    '工业品': ['螺纹钢', '热轧卷板', '线材', '不锈钢', '铜', '铝', '锌', '铅', '镍', '锡', '黄金', '白银',
              '纸浆', '氧化铝', '丁二烯橡胶', '国际铜', '集运指数', '工业硅', '碳酸锂', '多晶硅'],
    '化工品': ['塑料', 'PVC', 'PP', '乙二醇', '苯乙烯', '液化石油气', 'PTA', '甲醇', '玻璃', '尿素',
              '纯碱', '短纤', '烧碱', '对二甲苯', '瓶片'],
    '能源品': ['燃料油', '沥青', '天然橡胶', '原油', '低硫燃料油', '20号胶'],
    '黑色系': ['铁矿石', '焦炭', '焦煤', '硅铁', '锰硅'],
    '金融期货': ['沪深300', '上证50', '中证500', '中证1000', '5年期国债', '2年期国债']
}

def generate_demo_data(symbol, start_date, end_date):
    """生成模拟数据用于演示"""
    try:
        # 设置时间范围
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime.now()
        
        # 生成日期序列（只包含工作日）
        dates = pd.bdate_range(start=start_date, end=end_date, freq='B')
        
        # 生成模拟价格数据
        np.random.seed(hash(symbol) % 2**32)  # 基于品种名称设置随机种子，确保一致性
        
        # 基础价格
        base_price = 3000 + (hash(symbol) % 1000)
        
        # 生成价格序列（带有季节性特征）
        prices = []
        current_price = base_price
        
        for i, date in enumerate(dates):
            # 添加季节性因子
            month = date.month
            seasonal_factor = 1.0
            
            # 不同品种的季节性模式
            if symbol in ['C0', 'A0', 'B0']:  # 农产品
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (month - 3) / 12)
            elif symbol in ['RB0', 'HC0']:  # 钢材
                seasonal_factor = 1.0 + 0.05 * np.sin(2 * np.pi * (month - 1) / 12)
            
            # 随机波动
            daily_return = np.random.normal(0, 0.02) * seasonal_factor
            current_price = current_price * (1 + daily_return)
            prices.append(current_price)
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'close': prices
        })
        
        # 计算收益率
        df['return'] = df['close'].pct_change()
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df
        
    except Exception as e:
        st.error(f"生成模拟数据失败: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # 缓存1小时
def get_futures_data(symbol, start_date, end_date):
    """获取期货历史数据"""
    try:
        # 使用您之前成功的方法获取期货数据
        df = None
        
        # 方法1: 使用futures_zh_daily_sina（您之前成功的方法）
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
        except Exception as e:
            print(f"futures_zh_daily_sina失败: {e}")
        
        # 方法2: 如果第一种方法失败，尝试futures_main_sina
        if df is None or df.empty:
            try:
                df = ak.futures_main_sina(symbol=symbol)
            except Exception as e:
                print(f"futures_main_sina失败: {e}")
        
        # 方法3: 如果还是失败，尝试生成模拟数据用于演示
        if df is None or df.empty:
            st.warning(f"无法获取 {symbol} 的实时数据，正在生成模拟数据用于演示...")
            return generate_demo_data(symbol, start_date, end_date)
        
        if df is None or df.empty:
            return None
        
        # 检查数据列名并标准化
        if 'date' not in df.columns:
            # 尝试其他可能的日期列名
            date_columns = ['日期', 'date', 'Date', 'DATE', '时间']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col:
                df = df.rename(columns={date_col: 'date'})
            elif df.index.name and 'date' in str(df.index.name).lower():
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'date'})
            else:
                # 如果没有日期列，使用索引作为日期
                df = df.reset_index()
                if len(df.columns) > 0:
                    df = df.rename(columns={df.columns[0]: 'date'})
        
        # 检查价格列名并标准化
        price_columns = ['收盘价', 'close', 'Close', '收盘']
        close_col = None
        for col in price_columns:
            if col in df.columns:
                close_col = col
                break
        
        if close_col is None:
            # 如果没有找到收盘价列，使用数值列中的最后一列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                close_col = numeric_cols[-1]
            else:
                return None
        
        if close_col != 'close':
            df = df.rename(columns={close_col: 'close'})
            
        # 数据预处理
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date', 'close'])
        
        # 如果指定了时间范围，进行过滤
        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        
        df = df.sort_values('date').reset_index(drop=True)
        
        # 计算收益率
        df['return'] = df['close'].pct_change()
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        return None

def clean_price_data(df):
    """清理价格数据中的异常值"""
    if df.empty:
        return df
    
    df = df.copy()
    
    # 计算价格变化率
    df['price_change_pct'] = df['close'].pct_change()
    
    # 识别异常价格跳跃（超过50%的单日变化）
    extreme_changes = abs(df['price_change_pct']) > 0.5
    
    # 使用session state来收集处理信息
    if 'processing_info' not in st.session_state:
        st.session_state.processing_info = []
    
    if extreme_changes.any():
        jump_count = extreme_changes.sum()
        st.session_state.processing_info.append(f"发现 {jump_count} 个异常价格跳跃，已自动修正")
        
        # 对于异常跳跃，使用前一日价格进行插值修正
        for idx in df[extreme_changes].index:
            if idx > 0:
                old_price = df.loc[idx, 'close']
                df.loc[idx, 'close'] = df.loc[idx-1, 'close']
                date_str = df.loc[idx, 'date'].strftime('%Y-%m-%d')
                st.session_state.processing_info.append(f"修正 {date_str}: {old_price:.2f} → {df.loc[idx, 'close']:.2f}")
    
    # 重新计算收益率
    df['return'] = df['close'].pct_change()
    
    return df

def calculate_monthly_returns(df):
    """计算月度收益率统计，包含异常值处理"""
    if df.empty:
        return pd.DataFrame()
    
    # 清理数据
    df = clean_price_data(df)
    
    # 过滤极端收益率（超过20%的单日收益率视为异常）
    df_clean = df[abs(df['return']) <= 0.2].copy()
    
    # 使用session state来收集处理信息，避免重复显示
    if 'processing_info' not in st.session_state:
        st.session_state.processing_info = []
    
    if len(df_clean) < len(df):
        removed_count = len(df) - len(df_clean)
        st.session_state.processing_info.append(f"已过滤 {removed_count} 个极端收益率数据点")
    
    df_clean['month'] = df_clean['date'].dt.month
    
    # 计算月度统计
    monthly_stats = df_clean.groupby('month').agg({
        'return': ['mean', 'std', 'count', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
    })
    
    monthly_stats.columns = ['平均收益率', '收益率标准差', '交易天数', '上涨概率']
    
    # 计算t统计量和p值
    from scipy import stats
    monthly_stats['t统计量'] = monthly_stats['平均收益率'] / (monthly_stats['收益率标准差'] / np.sqrt(monthly_stats['交易天数']))
    monthly_stats['p值'] = 2 * (1 - stats.t.cdf(np.abs(monthly_stats['t统计量']), monthly_stats['交易天数'] - 1))
    
    # 显著性标记
    monthly_stats['显著性'] = monthly_stats['p值'].apply(
        lambda x: '***' if x < 0.01 else '**' if x < 0.05 else '*' if x < 0.1 else ''
    )
    
    # 确保所有月份都存在
    all_months = pd.DataFrame(index=range(1, 13))
    monthly_stats = all_months.join(monthly_stats, how='left').fillna(0)
    
    return monthly_stats

def create_heatmap(monthly_data, title="月度收益率热力图"):
    """创建改进的热力图"""
    if monthly_data.empty:
        return None
    
    # 准备数据矩阵
    symbols = monthly_data.index.tolist()
    months = ['1月', '2月', '3月', '4月', '5月', '6月', 
              '7月', '8月', '9月', '10月', '11月', '12月']
    
    # 创建数据矩阵
    data_matrix = []
    for symbol in symbols:
        row = []
        for month in range(1, 13):
            if month in monthly_data.columns:
                value = monthly_data.loc[symbol, month]
                # 确保是数值类型
                if pd.isna(value) or not isinstance(value, (int, float)):
                    value = 0.0
                row.append(float(value))
            else:
                row.append(0.0)
        data_matrix.append(row)
    
    # 转换为numpy数组
    data_matrix = np.array(data_matrix)
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=data_matrix,
        x=months,
        y=symbols,
        colorscale='RdBu_r',  # 红色为正，蓝色为负
        zmid=0,  # 设置颜色中心点为0
        text=[[f'{val:.3%}' for val in row] for row in data_matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='品种: %{y}<br>月份: %{x}<br>收益率: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="月份",
        yaxis_title="期货品种",
        height=max(400, len(symbols) * 25),
        font=dict(size=12)
    )
    
    return fig

def create_single_heatmap(monthly_stats, symbol_name):
    """创建单品种月度统计热力图"""
    if monthly_stats.empty:
        return None
    
    months = ['1月', '2月', '3月', '4月', '5月', '6月', 
              '7月', '8月', '9月', '10月', '11月', '12月']
    
    # 准备数据
    returns = [monthly_stats.loc[i, '平均收益率'] if i in monthly_stats.index else 0 
               for i in range(1, 13)]
    win_rates = [monthly_stats.loc[i, '上涨概率'] if i in monthly_stats.index else 0 
                 for i in range(1, 13)]
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['月度平均收益率', '月度上涨概率'],
        vertical_spacing=0.15
    )
    
    # 收益率热力图
    fig.add_trace(
        go.Heatmap(
            z=[returns],
            x=months,
            y=[symbol_name],
            colorscale='RdBu_r',
            zmid=0,
            text=[[f'{val:.3%}' for val in returns]],
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True,
            colorbar=dict(x=1.02, len=0.4, y=0.75),
            hovertemplate='月份: %{x}<br>收益率: %{text}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 胜率热力图
    fig.add_trace(
        go.Heatmap(
            z=[win_rates],
            x=months,
            y=[symbol_name],
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            text=[[f'{val:.1%}' for val in win_rates]],
            texttemplate='%{text}',
            textfont={"size": 12},
            showscale=True,
            colorbar=dict(x=1.02, len=0.4, y=0.25),
            hovertemplate='月份: %{x}<br>胜率: %{text}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f'{symbol_name} 月度季节性分析',
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_win_rate_chart(monthly_stats):
    """创建胜率柱状图"""
    if monthly_stats is None:
        return None
    
    months = ['1月', '2月', '3月', '4月', '5月', '6月', 
              '7月', '8月', '9月', '10月', '11月', '12月']
    
    fig = go.Figure(data=[
        go.Bar(
            x=months,
            y=monthly_stats['上涨概率'].values,
            text=[f'{val:.1%}' for val in monthly_stats['上涨概率'].values],
            textposition='auto',
            marker_color=['green' if x > 0.5 else 'red' for x in monthly_stats['上涨概率'].values]
        )
    ])
    
    fig.update_layout(
        title='月度上涨概率',
        xaxis_title='月份',
        yaxis_title='上涨概率',
        yaxis=dict(tickformat='.0%'),
        height=400
    )
    
    fig.add_hline(y=0.5, line_dash="dash", line_color="black", 
                  annotation_text="50%基准线")
    
    return fig

def create_return_distribution(df):
    """创建收益率分布图"""
    if df is None or df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('月度收益率箱线图', '月度收益率小提琴图', 
                       '收益率分布直方图', '累计收益率曲线'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 箱线图
    for month in range(1, 13):
        month_data = df[df['month'] == month]['return'].dropna()
        if not month_data.empty:
            fig.add_trace(
                go.Box(y=month_data, name=f'{month}月', showlegend=False),
                row=1, col=1
            )
    
    # 小提琴图
    for month in range(1, 13):
        month_data = df[df['month'] == month]['return'].dropna()
        if not month_data.empty:
            fig.add_trace(
                go.Violin(y=month_data, name=f'{month}月', showlegend=False),
                row=1, col=2
            )
    
    # 直方图
    fig.add_trace(
        go.Histogram(x=df['return'].dropna(), nbinsx=50, showlegend=False),
        row=2, col=1
    )
    
    # 累计收益率
    df_sorted = df.sort_values('date')
    cumulative_return = (1 + df_sorted['return'].fillna(0)).cumprod()
    fig.add_trace(
        go.Scatter(x=df_sorted['date'], y=cumulative_return, 
                  mode='lines', name='累计收益率', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="收益率分析图表")
    return fig

def generate_trading_signals(monthly_stats):
    """生成交易信号"""
    if monthly_stats is None or monthly_stats.empty:
        return None
    
    signals = []
    for month in range(1, 13):
        if month not in monthly_stats.index:
            continue
            
        stats = monthly_stats.loc[month]
        
        # 信号强度评分
        score = 0
        signal_type = "观望"
        
        # 基于平均收益率
        if stats['平均收益率'] > 0.01:  # 1%以上
            score += 30
        elif stats['平均收益率'] < -0.01:  # -1%以下
            score -= 30
            
        # 基于胜率
        if stats['上涨概率'] > 0.6:  # 60%以上
            score += 25
        elif stats['上涨概率'] < 0.4:  # 40%以下
            score -= 25
            
        # 基于稳定性（标准差）
        if stats['收益率标准差'] < 0.02:  # 2%以下
            score += 15
            
        # 确定信号类型
        if score >= 50:
            signal_type = "强烈看多"
        elif score >= 30:
            signal_type = "看多"
        elif score <= -50:
            signal_type = "强烈看空"
        elif score <= -30:
            signal_type = "看空"
            
        signals.append({
            '月份': f'{month}月',
            '平均收益率': f'{stats["平均收益率"]:.3%}',
            '上涨概率': f'{stats["上涨概率"]:.1%}',
            '收益率标准差': f'{stats["收益率标准差"]:.3%}',
            '交易天数': int(stats['交易天数']),
            '信号强度': score,
            '交易建议': signal_type
        })
    
    return pd.DataFrame(signals)

def display_monthly_ranking(all_results):
    """显示月度收益率排名，修复精度问题"""
    if not all_results:
        st.warning("没有数据可显示")
        return
    
    st.subheader("📊 月度收益率排名分析")
    
    # 选择月份
    selected_month = st.selectbox(
        "选择月份",
        options=list(range(1, 13)),
        format_func=lambda x: f"{x}月",
        key="ranking_month"
    )
    
    # 收集该月份的数据
    month_data = []
    for symbol, result in all_results.items():
        if 'stats' in result and selected_month in result['stats'].index:
            stats = result['stats'].loc[selected_month]
            month_data.append({
                '品种': symbol,
                '平均收益率': stats['平均收益率'],
                '收益率标准差': stats['收益率标准差'],
                '交易天数': int(stats['交易天数']),
                '上涨概率': stats['上涨概率']
            })
    
    if not month_data:
        st.warning(f"{selected_month}月没有可用数据")
        return
    
    # 创建DataFrame并排序
    df_ranking = pd.DataFrame(month_data)
    df_ranking = df_ranking.sort_values('平均收益率', ascending=False)
    
    # 格式化显示
    df_display = df_ranking.copy()
    df_display['平均收益率'] = df_display['平均收益率'].apply(lambda x: f"{x:.3%}")  # 3位小数
    df_display['收益率标准差'] = df_display['收益率标准差'].apply(lambda x: f"{x:.3%}")
    df_display['上涨概率'] = df_display['上涨概率'].apply(lambda x: f"{x:.1%}")
    
    st.dataframe(df_display, use_container_width=True)
    
    # 创建排名图表
    fig = go.Figure()
    
    # 添加收益率柱状图
    colors = ['red' if x >= 0 else 'blue' for x in df_ranking['平均收益率']]
    
    fig.add_trace(go.Bar(
        x=df_ranking['品种'],
        y=df_ranking['平均收益率'],
        marker_color=colors,
        text=[f'{x:.3%}' for x in df_ranking['平均收益率']],  # 3位小数
        textposition='outside',
        hovertemplate='品种: %{x}<br>收益率: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{selected_month}月各品种平均收益率排名',
        xaxis_title='期货品种',
        yaxis_title='平均收益率',
        yaxis_tickformat='.3%',  # Y轴显示3位小数
        height=500,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)

def generate_professional_analysis(results, significant_patterns, monthly_summary):
    """生成专业的投资分析总结"""
    st.markdown("### 📊 专业投资分析报告")
    
    # 分析品种分类
    agricultural_products = []
    industrial_products = []
    energy_products = []
    financial_products = []
    
    # 根据品种名称分类
    for symbol in results.keys():
        if symbol in ['玉米', '大豆一号', '大豆二号', '豆粕', '豆油', '棕榈油', '菜籽油', '菜籽粕', '白糖', '棉花', '苹果', '红枣', '花生', '淀粉']:
            agricultural_products.append(symbol)
        elif symbol in ['螺纹钢', '热轧卷板', '不锈钢', '线材', '铁矿石', '焦炭', '焦煤', '动力煤', '铜', '铝', '锌', '铅', '镍', '锡', '黄金', '白银']:
            industrial_products.append(symbol)
        elif symbol in ['原油', '燃料油', '石油沥青', '天然气', '液化石油气']:
            energy_products.append(symbol)
        elif symbol in ['沪深300', '中证500', '上证50', '10年期国债']:
            financial_products.append(symbol)
    
    # 生成分类分析
    st.markdown("#### 🌾 农产品季节性规律分析")
    if agricultural_products:
        analyze_category_seasonality(results, agricultural_products, "农产品", 
                                   "农产品价格主要受种植、收获周期影响，呈现明显的季节性特征")
    
    st.markdown("#### 🏭 工业品季节性规律分析")
    if industrial_products:
        analyze_category_seasonality(results, industrial_products, "工业品",
                                   "工业品价格受宏观经济周期、基建投资节奏和环保政策影响")
    
    st.markdown("#### ⛽ 能源品季节性规律分析")
    if energy_products:
        analyze_category_seasonality(results, energy_products, "能源品",
                                   "能源品价格受季节性需求变化、地缘政治和库存周期影响")
    
    # 整体投资策略建议
    st.markdown("#### 💡 投资策略建议")
    
    if significant_patterns:
        # 按月份分组显著模式
        monthly_patterns = {}
        for pattern in significant_patterns:
            month = pattern['月份']
            if month not in monthly_patterns:
                monthly_patterns[month] = []
            monthly_patterns[month].append(pattern)
        
        # 生成月度投资建议
        investment_recommendations = []
        
        for month in range(1, 13):
            if month in monthly_patterns:
                patterns = monthly_patterns[month]
                positive_patterns = [p for p in patterns if p['收益率'] > 0]
                negative_patterns = [p for p in patterns if p['收益率'] < 0]
                
                if positive_patterns:
                    best_pattern = max(positive_patterns, key=lambda x: x['收益率'])
                    investment_recommendations.append({
                        '月份': f"{month}月",
                        '推荐操作': '做多',
                        '推荐品种': best_pattern['品种'],
                        '预期收益': f"{best_pattern['收益率']:.3%}",
                        '统计显著性': best_pattern['显著性'],
                        'p值': f"{best_pattern['p值']:.3f}"
                    })
                elif negative_patterns:
                    worst_pattern = min(negative_patterns, key=lambda x: x['收益率'])
                    investment_recommendations.append({
                        '月份': f"{month}月",
                        '推荐操作': '做空',
                        '推荐品种': worst_pattern['品种'],
                        '预期收益': f"{abs(worst_pattern['收益率']):.3%}",
                        '统计显著性': worst_pattern['显著性'],
                        'p值': f"{worst_pattern['p值']:.3f}"
                    })
        
        if investment_recommendations:
            st.markdown("**基于统计显著性的月度投资建议：**")
            recommendations_df = pd.DataFrame(investment_recommendations)
            st.dataframe(recommendations_df, use_container_width=True)
            
            # 下载投资建议
            csv_recommendations = recommendations_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载投资建议 (CSV)",
                data=csv_recommendations,
                file_name="月度投资建议.csv",
                mime="text/csv",
                key="download_investment_recommendations"
            )
    
    # 风险提示和投资原则
    st.markdown("#### ⚠️ 风险提示与投资原则")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **风险提示：**
        - 历史季节性规律不保证未来重现
        - 宏观经济环境变化可能改变季节性特征
        - 极端天气、政策变化等因素可能打破常规
        - 统计显著性不等于投资收益保证
        - 期货交易具有高杠杆风险
        """)
    
    with col2:
        st.markdown("""
        **投资原则：**
        - 严格设置止损，控制单笔损失
        - 合理配置仓位，避免过度集中
        - 结合基本面分析验证季节性信号
        - 关注技术面确认入场时机
        - 保持理性，避免情绪化交易
        """)
    
    # 市场展望
    st.markdown("#### 🔮 市场展望")
    
    if monthly_summary:
        summary_df = pd.DataFrame(monthly_summary).T
        best_months = summary_df.nlargest(3, '平均收益率').index.tolist()
        worst_months = summary_df.nsmallest(3, '平均收益率').index.tolist()
        
        st.markdown(f"""
        **季节性交易机会展望：**
        
        **最佳交易月份：** {', '.join([f'{m}月' for m in best_months])}
        - 这些月份历史上表现较好，可重点关注做多机会
        - 建议提前布局，把握季节性上涨行情
        
        **谨慎交易月份：** {', '.join([f'{m}月' for m in worst_months])}
        - 这些月份历史表现相对较弱，需谨慎操作
        - 可考虑做空策略或减少仓位配置
        
        **投资建议：**
        1. 建立季节性交易日历，提前规划投资策略
        2. 结合宏观经济数据和政策变化调整预期
        3. 采用组合投资方式，分散单一品种风险
        4. 定期回顾和优化交易策略，适应市场变化
        """)

def analyze_category_seasonality(results, category_symbols, category_name, description):
    """分析特定品类的季节性规律"""
    if not category_symbols:
        return
    
    st.markdown(f"**{category_name}分析结果：**")
    
    # 收集该品类的季节性数据
    category_patterns = []
    category_monthly_data = {}
    
    for symbol in category_symbols:
        if symbol in results and 'stats' in results[symbol]:
            stats = results[symbol]['stats']
            
            # 找出显著月份
            significant_months = stats[stats['p值'] < 0.05]
            for month, row in significant_months.iterrows():
                category_patterns.append({
                    '品种': symbol,
                    '月份': month,
                    '收益率': row['平均收益率'],
                    'p值': row['p值']
                })
            
            # 收集月度数据
            for month in range(1, 13):
                if month in stats.index:
                    if month not in category_monthly_data:
                        category_monthly_data[month] = []
                    category_monthly_data[month].append(stats.loc[month, '平均收益率'])
    
    if category_patterns:
        # 按月份分组
        monthly_analysis = {}
        for pattern in category_patterns:
            month = pattern['月份']
            if month not in monthly_analysis:
                monthly_analysis[month] = {'positive': [], 'negative': []}
            
            if pattern['收益率'] > 0:
                monthly_analysis[month]['positive'].append(pattern)
            else:
                monthly_analysis[month]['negative'].append(pattern)
        
        # 生成分析结论
        strong_months = []
        weak_months = []
        
        for month, patterns in monthly_analysis.items():
            if len(patterns['positive']) > len(patterns['negative']):
                strong_months.append(month)
            elif len(patterns['negative']) > len(patterns['positive']):
                weak_months.append(month)
        
        if strong_months or weak_months:
            analysis_text = f"{description}。\n\n"
            
            if strong_months:
                analysis_text += f"**表现较强月份：** {', '.join([f'{m}月' for m in sorted(strong_months)])}\n"
            
            if weak_months:
                analysis_text += f"**表现较弱月份：** {', '.join([f'{m}月' for m in sorted(weak_months)])}\n"
            
            # 添加具体的季节性解释
            if category_name == "农产品":
                analysis_text += """
**季节性原因分析：**
- 春季（3-5月）：播种期，天气炒作推动价格上涨
- 夏季（6-8月）：生长期，天气风险关注度高
- 秋季（9-11月）：收获期，供应增加压制价格
- 冬季（12-2月）：消费旺季，节日需求支撑价格
                """
            elif category_name == "工业品":
                analysis_text += """
**季节性原因分析：**
- 春季（3-5月）：基建复工，需求回升推动价格
- 夏季（6-8月）：高温多雨影响施工，需求相对平稳
- 秋季（9-11月）：施工黄金期，需求旺盛
- 冬季（12-2月）：环保限产，供应收缩支撑价格
                """
            elif category_name == "能源品":
                analysis_text += """
**季节性原因分析：**
- 春季（3-5月）：炼厂检修季，供应偏紧
- 夏季（6-8月）：驾驶旺季，汽油需求增加
- 秋季（9-11月）：冬储行情，取暖需求预期
- 冬季（12-2月）：取暖高峰，能源消费旺季
                """
            
            st.markdown(analysis_text)
    else:
        st.info(f"{category_name}暂无统计显著的季节性模式")

# 主界面
def main():
    st.markdown('<h1 class="main-header">📈 期货季节性交易规律分析系统</h1>', unsafe_allow_html=True)
    
    # 作者信息
    st.markdown("""
    <div style="text-align: center; padding: 10px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;">
        <p style="margin: 0; color: #666;">
            <strong>作者：7haoge</strong> | 
            <strong>邮箱：953534947@qq.com</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏配置
    st.sidebar.markdown("## 📊 分析配置")
    
    # 品种选择
    selection_type = st.sidebar.radio(
        "选择方式",
        ["按分类选择", "手动选择品种", "全部品种"]
    )
    
    selected_symbols = []
    
    if selection_type == "按分类选择":
        selected_categories = st.sidebar.multiselect(
            "选择品种分类",
            list(CATEGORY_MAPPING.keys()),
            default=["农产品"]
        )
        for category in selected_categories:
            selected_symbols.extend(CATEGORY_MAPPING[category])
            
    elif selection_type == "手动选择品种":
        selected_symbols = st.sidebar.multiselect(
            "选择期货品种",
            list(FUTURES_MAPPING.keys()),
            default=["玉米", "大豆一号", "白糖"]
        )
    else:
        selected_symbols = list(FUTURES_MAPPING.keys())
    
    # 时间范围选择
    st.sidebar.markdown("### 📅 时间范围")
    time_range = st.sidebar.radio(
        "选择时间范围",
        ["全部时间段", "自定义范围"]
    )
    
    if time_range == "全部时间段":
        start_date = None
        end_date = None
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=datetime(2020, 1, 1))
        with col2:
            end_date = st.date_input("结束日期", value=datetime.now())
    
    # 初始化会话状态
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    
    # 分析按钮
    if st.sidebar.button("🚀 开始分析", type="primary"):
        if not selected_symbols:
            st.error("请至少选择一个期货品种！")
            return
            
        st.markdown('<div class="sub-header">🔄 正在获取数据和分析...</div>', unsafe_allow_html=True)
        
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = {}
        total_symbols = len(selected_symbols)
        
        for i, symbol in enumerate(selected_symbols):
            status_text.text(f'正在分析 {symbol} ({i+1}/{total_symbols})')
            progress_bar.progress((i + 1) / total_symbols)
            
            if symbol in FUTURES_MAPPING:
                symbol_code = FUTURES_MAPPING[symbol]
                df = get_futures_data(symbol_code, start_date, end_date)
                
                if df is not None and not df.empty:
                    monthly_stats = calculate_monthly_returns(df)
                    if monthly_stats is not None:
                        results[symbol] = {
                            'data': df,
                            'stats': monthly_stats,
                            'symbol_code': symbol_code
                        }
        
        progress_bar.empty()
        status_text.empty()
        
        # 显示数据处理信息（折叠形式）
        if hasattr(st.session_state, 'processing_info') and st.session_state.processing_info:
            with st.expander("📋 数据处理日志", expanded=False):
                st.markdown("**数据清理和异常处理记录：**")
                
                # 按类型分组显示
                price_fixes = [info for info in st.session_state.processing_info if "异常价格跳跃" in info]
                return_filters = [info for info in st.session_state.processing_info if "极端收益率" in info]
                price_corrections = [info for info in st.session_state.processing_info if "修正" in info and "→" in info]
                
                if price_fixes:
                    st.warning("⚠️ 异常价格跳跃处理：")
                    for info in price_fixes:
                        st.write(f"  • {info}")
                
                if return_filters:
                    st.info("🔍 极端收益率过滤：")
                    for info in return_filters:
                        st.write(f"  • {info}")
                
                if price_corrections:
                    st.info("🔧 详细价格修正记录：")
                    with st.container():
                        for info in price_corrections:
                            st.write(f"  • {info}")
                
                st.markdown("""
                **处理说明：**
                - **异常价格跳跃**：单日价格变化超过50%被视为数据错误，使用前一日价格替代
                - **极端收益率**：单日收益率超过±20%被视为异常值，直接过滤不参与统计
                - **科学依据**：期货价格通常不会出现如此极端的单日变化，这些多为数据错误或特殊事件
                - **统计影响**：这些处理确保了月度统计的稳健性和可靠性
                """)
        
        if not results:
            st.error("未能获取到有效数据，请检查网络连接或稍后重试。")
            return
        
        # 保存结果到会话状态
        st.session_state.analysis_results = results
        st.session_state.analysis_completed = True
        
        # 清空处理信息，避免下次分析时重复显示
        st.session_state.processing_info = []
    
    # 显示分析结果（无论是新分析还是已保存的结果）
    if st.session_state.analysis_completed and st.session_state.analysis_results:
        
        results = st.session_state.analysis_results
        
        # 显示分析结果
        st.markdown('<div class="sub-header">📈 分析结果</div>', unsafe_allow_html=True)
        
        # 概览信息
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("分析品种数", len(results))
        with col2:
            total_days = sum([len(r['data']) for r in results.values()])
            st.metric("总交易天数", f"{total_days:,}")
        with col3:
            avg_return = np.mean([r['stats']['平均收益率'].mean() for r in results.values()])
            st.metric("平均月收益率", f"{avg_return:.2%}")
        with col4:
            significant_months = sum([
                (r['stats']['p值'] < 0.05).sum() for r in results.values()
            ])
            st.metric("显著月份数", significant_months)
        
        # 品种选择器（用于详细分析）
        st.markdown("### 🔍 详细分析")
        selected_for_detail = st.selectbox(
            "选择品种进行详细分析",
            list(results.keys()),
            key="detail_selector"  # 添加唯一key避免重新加载
        )
        
        if selected_for_detail and selected_for_detail in results:
            result = results[selected_for_detail]
            df = result['data']
            monthly_stats = result['stats']
            
            # 创建标签页
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 统计概览", "🔥 热力图", "📈 图表分析", "🎯 交易信号", "📋 详细数据"
            ])
            
            with tab1:
                st.markdown(f"#### {selected_for_detail} 季节性统计概览")
                
                # 添加统计指标说明
                with st.expander("📖 统计指标说明"):
                    st.markdown("""
                    **月度统计指标解释：**
                    
                    - **平均收益率**: 该月份所有交易日收益率的算术平均值
                    - **收益率标准差**: 衡量该月份收益率的波动程度，数值越大表示波动越大
                    - **交易天数**: 该月份的有效交易日数量
                    - **上涨概率**: 该月份收益率为正的交易日占比（胜率）
                    - **t统计量**: 用于检验平均收益率是否显著不为0的统计量
                      - 计算公式: t = 平均收益率 / (标准差 / √交易天数)
                      - |t| > 2 通常表示统计显著
                    - **p值**: 统计显著性概率，表示观察到当前结果的概率
                      - p < 0.01: 高度显著 (***)
                      - p < 0.05: 显著 (**)  
                      - p < 0.1: 边际显著 (*)
                      - p ≥ 0.1: 不显著
                    - **显著性**: 基于p值的显著性标记
                    
                    **收益率计算方法：**
                    - 日收益率 = (今日收盘价 - 昨日收盘价) / 昨日收盘价
                    - 月度平均收益率 = 该月所有日收益率的算术平均值
                    - 这种方法能够消除价格水平差异，便于不同时期和品种的比较
                    """)
                
                
                # 关键指标
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**最佳交易月份（按平均收益率）**")
                    best_months = monthly_stats.nlargest(3, '平均收益率')
                    for idx, (month, row) in enumerate(best_months.iterrows()):
                        st.markdown(f"{idx+1}. {month}月: {row['平均收益率']:.2%} {row['显著性']}")
                
                with col2:
                    st.markdown("**最差交易月份（按平均收益率）**")
                    worst_months = monthly_stats.nsmallest(3, '平均收益率')
                    for idx, (month, row) in enumerate(worst_months.iterrows()):
                        st.markdown(f"{idx+1}. {month}月: {row['平均收益率']:.2%} {row['显著性']}")
                
                # 统计表格
                st.markdown("**月度统计数据**")
                display_stats = monthly_stats.copy()
                display_stats['平均收益率'] = display_stats['平均收益率'].apply(lambda x: f"{x:.2%}")
                display_stats['收益率标准差'] = display_stats['收益率标准差'].apply(lambda x: f"{x:.2%}")
                display_stats['上涨概率'] = display_stats['上涨概率'].apply(lambda x: f"{x:.1%}")
                display_stats['p值'] = display_stats['p值'].apply(lambda x: f"{x:.3f}")
                st.dataframe(display_stats, use_container_width=True)
                
                # 添加统计数据下载
                col1, col2 = st.columns(2)
                with col1:
                    csv_stats = monthly_stats.to_csv(encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载统计数据 (CSV)",
                        data=csv_stats,
                        file_name=f"{selected_for_detail}_月度统计.csv",
                        mime="text/csv"
                    )
                with col2:
                    # Excel格式下载
                    try:
                        import io
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            monthly_stats.to_excel(writer, sheet_name='月度统计', index=True)
                        excel_data = excel_buffer.getvalue()
                        st.download_button(
                            label="📥 下载统计数据 (Excel)",
                            data=excel_data,
                            file_name=f"{selected_for_detail}_月度统计.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except ImportError:
                        st.error("Excel导出需要安装openpyxl包: pip install openpyxl")
            
            with tab2:
                st.markdown(f"#### {selected_for_detail} 季节性热力图")
                heatmap_fig = create_single_heatmap(monthly_stats, selected_for_detail)
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                    
                    # 添加热力图下载选项
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 下载热力图 (HTML)", key="download_heatmap_html"):
                            heatmap_html = heatmap_fig.to_html()
                            st.download_button(
                                label="下载热力图HTML文件",
                                data=heatmap_html,
                                file_name=f"{selected_for_detail}_热力图.html",
                                mime="text/html"
                            )
                    with col2:
                        if st.button("📥 下载热力图 (PNG)", key="download_heatmap_png"):
                            try:
                                img_bytes = heatmap_fig.to_image(format="png", width=800, height=400)
                                st.download_button(
                                    label="下载热力图PNG文件",
                                    data=img_bytes,
                                    file_name=f"{selected_for_detail}_热力图.png",
                                    mime="image/png"
                                )
                            except:
                                st.error("PNG下载需要安装kaleido包: pip install kaleido")
                
                # 胜率图表
                win_rate_fig = create_win_rate_chart(monthly_stats)
                if win_rate_fig:
                    st.plotly_chart(win_rate_fig, use_container_width=True)
                    
                    # 添加胜率图下载选项
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 下载胜率图 (HTML)", key="download_winrate_html"):
                            winrate_html = win_rate_fig.to_html()
                            st.download_button(
                                label="下载胜率图HTML文件",
                                data=winrate_html,
                                file_name=f"{selected_for_detail}_胜率图.html",
                                mime="text/html"
                            )
                    with col2:
                        if st.button("📥 下载胜率图 (PNG)", key="download_winrate_png"):
                            try:
                                img_bytes = win_rate_fig.to_image(format="png", width=800, height=400)
                                st.download_button(
                                    label="下载胜率图PNG文件",
                                    data=img_bytes,
                                    file_name=f"{selected_for_detail}_胜率图.png",
                                    mime="image/png"
                                )
                            except:
                                st.error("PNG下载需要安装kaleido包: pip install kaleido")
            
            with tab3:
                st.markdown(f"#### {selected_for_detail} 详细图表分析")
                distribution_fig = create_return_distribution(df)
                if distribution_fig:
                    st.plotly_chart(distribution_fig, use_container_width=True)
                    
                    # 添加分布图下载选项
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 下载分布图 (HTML)", key="download_distribution_html"):
                            distribution_html = distribution_fig.to_html()
                            st.download_button(
                                label="下载分布图HTML文件",
                                data=distribution_html,
                                file_name=f"{selected_for_detail}_分布图.html",
                                mime="text/html"
                            )
                    with col2:
                        if st.button("📥 下载分布图 (PNG)", key="download_distribution_png"):
                            try:
                                img_bytes = distribution_fig.to_image(format="png", width=1200, height=800)
                                st.download_button(
                                    label="下载分布图PNG文件",
                                    data=img_bytes,
                                    file_name=f"{selected_for_detail}_分布图.png",
                                    mime="image/png"
                                )
                            except:
                                st.error("PNG下载需要安装kaleido包: pip install kaleido")
            
            with tab4:
                st.markdown(f"#### {selected_for_detail} 交易信号")
                signals_df = generate_trading_signals(monthly_stats)
                if signals_df is not None:
                    # 信号强度颜色映射
                    def color_signal(val):
                        if "强烈看多" in val:
                            return 'background-color: #d4edda; color: #155724'
                        elif "看多" in val:
                            return 'background-color: #d1ecf1; color: #0c5460'
                        elif "强烈看空" in val:
                            return 'background-color: #f8d7da; color: #721c24'
                        elif "看空" in val:
                            return 'background-color: #ffeaa7; color: #856404'
                        else:
                            return 'background-color: #f8f9fa; color: #495057'
                    
                    styled_df = signals_df.style.applymap(color_signal, subset=['交易建议'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # 交易建议总结
                    st.markdown("**交易建议总结**")
                    strong_long = signals_df[signals_df['交易建议'] == '强烈看多']['月份'].tolist()
                    long_signals = signals_df[signals_df['交易建议'] == '看多']['月份'].tolist()
                    strong_short = signals_df[signals_df['交易建议'] == '强烈看空']['月份'].tolist()
                    short_signals = signals_df[signals_df['交易建议'] == '看空']['月份'].tolist()
                    
                    if strong_long:
                        st.success(f"🚀 强烈看多月份: {', '.join(strong_long)}")
                    if long_signals:
                        st.info(f"📈 看多月份: {', '.join(long_signals)}")
                    if strong_short:
                        st.error(f"🔻 强烈看空月份: {', '.join(strong_short)}")
                    if short_signals:
                        st.warning(f"📉 看空月份: {', '.join(short_signals)}")
            
            with tab5:
                st.markdown(f"#### {selected_for_detail} 原始数据")
                st.dataframe(df.head(100), use_container_width=True)
                
                # 数据下载
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 下载完整数据 (CSV)",
                    data=csv,
                    file_name=f"{selected_for_detail}_历史数据.csv",
                    mime="text/csv"
                )
        
        # 综合分析
        if len(results) > 1:
            st.markdown("### 🌟 综合季节性分析")
            
            # 创建综合热力图数据
            monthly_data = pd.DataFrame(columns=range(1, 13))  # 预先设置列名
            
            for symbol, result in results.items():
                if 'stats' in result:
                    # 提取月度收益率数据
                    symbol_returns = []
                    for month in range(1, 13):
                        if month in result['stats'].index:
                            ret = result['stats'].loc[month, '平均收益率']
                            symbol_returns.append(ret)
                        else:
                            symbol_returns.append(0.0)
                    
                    # 添加到DataFrame
                    monthly_data.loc[symbol] = symbol_returns
            
            if not monthly_data.empty:
                # 创建综合热力图
                heatmap_fig = create_heatmap(monthly_data, "所有品种月度收益率综合热力图")
                if heatmap_fig:
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                # 添加综合热力图下载选项
                col1, col2 = st.columns(2)
                with col1:
                    # 直接使用download_button，避免嵌套按钮问题
                    combined_html = heatmap_fig.to_html()
                    st.download_button(
                        label="📥 下载综合热力图 (HTML)",
                        data=combined_html,
                        file_name="综合季节性热力图.html",
                        mime="text/html",
                        key="download_combined_heatmap_html"
                    )
                with col2:
                    try:
                        img_bytes = heatmap_fig.to_image(format="png", width=1200, height=max(600, len(monthly_data) * 30))
                        st.download_button(
                            label="📥 下载综合热力图 (PNG)",
                            data=img_bytes,
                            file_name="综合季节性热力图.png",
                            mime="image/png",
                            key="download_combined_heatmap_png"
                        )
                    except Exception as e:
                        st.error(f"PNG下载失败: {str(e)}")
                        st.info("提示：PNG下载需要安装kaleido包: pip install kaleido")
                
                # 月度排名
                display_monthly_ranking(results)
                
                # 添加综合分析总结
                st.markdown("### 📋 综合分析总结")
                
                # 计算整体统计
                all_monthly_returns = []
                significant_patterns = []
                
                for symbol, result in results.items():
                    if 'stats' in result:
                        stats = result['stats']
                        # 收集所有月度收益率
                        all_monthly_returns.extend(stats['平均收益率'].tolist())
                        
                        # 找出显著的季节性模式
                        significant_months = stats[stats['p值'] < 0.05]
                        for month, row in significant_months.iterrows():
                            significant_patterns.append({
                                '品种': symbol,
                                '月份': month,
                                '收益率': row['平均收益率'],
                                'p值': row['p值'],
                                '显著性': row['显著性']
                            })
                
                # 显示整体统计
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_return = np.mean(all_monthly_returns)
                    st.metric("整体平均月收益率", f"{avg_return:.3%}")
                
                with col2:
                    std_return = np.std(all_monthly_returns)
                    st.metric("收益率标准差", f"{std_return:.3%}")
                
                with col3:
                    st.metric("显著模式数量", len(significant_patterns))
                
                # 显著季节性模式
                if significant_patterns:
                    st.markdown("#### 🎯 发现的显著季节性模式")
                    patterns_df = pd.DataFrame(significant_patterns)
                    patterns_df = patterns_df.sort_values(['p值', '收益率'], ascending=[True, False])
                    
                    # 格式化显示
                    patterns_display = patterns_df.copy()
                    patterns_display['收益率'] = patterns_display['收益率'].apply(lambda x: f"{x:.3%}")
                    patterns_display['p值'] = patterns_display['p值'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(patterns_display, use_container_width=True)
                    
                    # 下载显著模式数据
                    csv_patterns = patterns_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 下载显著模式数据 (CSV)",
                        data=csv_patterns,
                        file_name="显著季节性模式.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("未发现统计显著的季节性模式")
                
                # 月度表现总结
                st.markdown("#### 📊 各月份整体表现")
                
                # 计算每个月份的平均表现
                monthly_summary = {}
                for month in range(1, 13):
                    month_returns = []
                    month_count = 0
                    significant_count = 0
                    
                    for symbol, result in results.items():
                        if 'stats' in result and month in result['stats'].index:
                            stats = result['stats'].loc[month]
                            month_returns.append(stats['平均收益率'])
                            month_count += 1
                            if stats['p值'] < 0.05:
                                significant_count += 1
                    
                    if month_returns:
                        monthly_summary[month] = {
                            '平均收益率': np.mean(month_returns),
                            '品种数量': month_count,
                            '显著品种数': significant_count,
                            '显著比例': significant_count / month_count if month_count > 0 else 0
                        }
                
                # 创建月度总结表格
                if monthly_summary:
                    summary_df = pd.DataFrame(monthly_summary).T
                    summary_df.index.name = '月份'
                    
                    # 格式化显示
                    summary_display = summary_df.copy()
                    summary_display['平均收益率'] = summary_display['平均收益率'].apply(lambda x: f"{x:.3%}")
                    summary_display['显著比例'] = summary_display['显著比例'].apply(lambda x: f"{x:.1%}")
                    
                    st.dataframe(summary_display, use_container_width=True)
                    
                    # 找出最佳和最差月份
                    best_month = summary_df['平均收益率'].idxmax()
                    worst_month = summary_df['平均收益率'].idxmin()
                    most_significant_month = summary_df['显著比例'].idxmax()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.success(f"🏆 最佳月份: {best_month}月 ({summary_df.loc[best_month, '平均收益率']:.3%})")
                    with col2:
                        st.error(f"📉 最差月份: {worst_month}月 ({summary_df.loc[worst_month, '平均收益率']:.3%})")
                    with col3:
                        st.info(f"🎯 最显著月份: {most_significant_month}月 ({summary_df.loc[most_significant_month, '显著比例']:.1%})")
                
                # 添加专业投资分析总结
                generate_professional_analysis(results, significant_patterns, monthly_summary)
            else:
                st.info("综合热力图数据为空，请确保选择的品种有有效的分析结果。")
    
    # 使用说明
    with st.expander("📖 使用说明"):
        st.markdown("""
        ### 系统功能说明
        
                 **1. 数据获取**
         - 支持按分类选择、手动选择或全部品种
         - 支持全部时间段或自定义时间范围分析
         - 数据来源：新浪财经期货数据（如无法获取则使用模拟数据演示）
        
        **2. 分析功能**
        - **统计概览**: 月度收益率、胜率、显著性检验
        - **热力图**: 直观显示季节性规律
        - **图表分析**: 收益率分布、累计收益等
        - **交易信号**: 基于统计显著性的交易建议
        - **详细数据**: 原始数据查看和下载
        
        **3. 信号解读**
        - **强烈看多/看空**: 统计显著且收益率较高
        - **看多/看空**: 有一定统计基础的信号
        - **观望**: 无明显季节性规律
        
        **4. 风险提示**
        - 历史规律不代表未来表现
        - 需结合基本面和技术面分析
        - 建议设置止损和仓位管理
        """)

if __name__ == "__main__":
    main()