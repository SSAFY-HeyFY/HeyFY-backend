import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler

def plot_overlapping_graphs(file_paths):
    """
    여러 개의 경제 지표 CSV 파일을 병합하지 않고, 하나의 그래프에 겹쳐서 시각화합니다.

    Args:
        file_paths (list): CSV 파일 경로의 리스트.
    """
    # 1. 데이터 불러오기 및 전처리
    data_frames = []
    for file in file_paths:
        df = pd.read_csv(file, encoding="CP949")
        # '날짜' 열을 datetime 객체로 변환
        df['날짜'] = pd.to_datetime(df['날짜'])

        # 숫자형으로 변환해야 할 열 리스트
        numeric_cols = ['종가', '시가', '고가', '저가']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
        
        # 날짜순으로 정렬
        df = df.sort_values(by='날짜')
        data_frames.append(df)

    # 2. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('경제 지표 비교 시각화', fontsize=16)
    
    plot_columns = ['종가', '시가', '고가', '저가']
    ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['blue', 'green', 'red'] # 파일별 색상 지정

    for col, (r, c) in zip(plot_columns, ax_indices):
        ax = axes[r, c]
        for i, df in enumerate(data_frames):
            ax.plot(df['날짜'], df[col], label=f'파일 {i+1}', color=colors[i])
        
        ax.set_title(f'{col} 변동 추이 비교')
        ax.set_xlabel('날짜')
        ax.set_ylabel('가격')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('economic_indicators_comparison.png')
    plt.close()

    print("경제 지표 비교 시각화 그래프를 'economic_indicators_comparison.png' 파일로 저장했습니다.")


def plot_custom_labeled_graphs(file_paths, labels):
    """
    (라벨 수정) 여러 개의 경제 지표 CSV 파일을 정규화하고,
    사용자 지정 라벨을 사용하여 하나의 그래프에 겹쳐서 시각화합니다.

    Args:
        file_paths (list): CSV 파일 경로의 리스트.
        labels (list): 그래프에 표시할 라벨의 리스트.
    """
    if len(file_paths) != len(labels):
        print("파일의 수와 라벨의 수가 일치하지 않습니다.")
        return

    # 1. 데이터 불러오기 및 전처리
    data_frames = []
    for file in file_paths:
        df = pd.read_csv(file, encoding="CP949")
        df['날짜'] = pd.to_datetime(df['날짜'])
        numeric_cols = ['종가', '시가', '고가', '저가']
        for col in numeric_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '').astype(float)
        df = df.sort_values(by='날짜')
        data_frames.append(df)

    # 2. 정규화
    normalized_dfs = []
    for df in data_frames:
        scaler = MinMaxScaler()
        normalized_df = df.copy()
        normalized_df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        normalized_dfs.append(normalized_df)

    # 3. 시각화 (수정된 부분)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('사용자 지정 라벨 경제 지표 비교', fontsize=16)
    
    plot_columns = ['종가', '시가', '고가', '저가']
    ax_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
    colors = ['blue', 'green', 'red', "orange"]

    for col, (r, c) in zip(plot_columns, ax_indices):
        ax = axes[r, c]
        for i, df in enumerate(normalized_dfs):
            # 사용자 지정 라벨을 사용
            ax.plot(df['날짜'], df[col], label=labels[i], color=colors[i])
        
        ax.set_title(f'정규화된 {col} 변동 추이 비교')
        ax.set_xlabel('날짜')
        ax.set_ylabel('정규화된 값')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('custom_labeled_economic_indicators.png')
    plt.close()

    print("사용자 지정 라벨 그래프를 'custom_labeled_economic_indicators.png' 파일로 저장했습니다.")



def main():
    # 한글 폰트 설정
    plt.rc('font', family='Malgun Gothic')

    file_paths = ['data/Investig_USD_KRW_20100101_20250816.csv', 'data/Investing_달러_지수_20100104_20250816.csv', 'data/Investing_미국10년물_국채_금리_채권수익률_20100104_20250816.csv', 'data/Investing_미국_달러_지수_선물_20100104_20250816.csv']
    labels = ["USD/KRW 환율", "미국달러지수", "미국10년물 국채 금리 채권수익률", "미국달러지수선물"]
    plot_custom_labeled_graphs(file_paths, labels)

if __name__ == "__main__":
    main()