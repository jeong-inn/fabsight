import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.process_map import get_process_info

def analyze_process_contribution():
    # 전체 SHAP 결과 로드 (top5만 있으면 top5로)
    top5 = pd.read_csv('data/raw/top5_sensors.csv')

    # 공정별 SHAP 합산
    process_scores = {}
    sensor_details = []

    for _, row in top5.iterrows():
        sid = int(row['sensor'])
        info = get_process_info(sid)
        process = info['process']
        score = float(row['shap_score'])

        process_scores[process] = process_scores.get(process, 0) + score
        sensor_details.append({
            'sensor_id': sid,
            'process': process,
            'stage': info['stage'],
            'shap_score': round(score, 4),
        })

    # 기여도 퍼센트 계산
    total = sum(process_scores.values())
    process_pct = {k: round(v / total * 100, 1) for k, v in process_scores.items()}

    # 출력
    print("=== 공정별 이상 기여도 ===")
    for p, pct in sorted(process_pct.items(), key=lambda x: -x[1]):
        print(f"  {p}: {pct}%")

    print("\n=== 센서별 상세 ===")
    df = pd.DataFrame(sensor_details)
    print(df.to_string(index=False))

    # 시각화
    COLORS = {'CVD': '#4a90d9', 'ETCH': '#d94a4a', 'CMP': '#4ad94a', 'LITHO': '#d9a84a'}
    processes = list(process_pct.keys())
    pcts = [process_pct[p] for p in processes]
    colors = [COLORS.get(p, '#888') for p in processes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Process Contribution to Anomaly Detection', fontsize=14, fontweight='bold')

    # 파이차트
    ax1.pie(pcts, labels=processes, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
    ax1.set_title('Process Anomaly Contribution')

    # 바차트 (센서별)
    df_sorted = df.sort_values('shap_score', ascending=True)
    bar_colors = [COLORS.get(p, '#888') for p in df_sorted['process']]
    bars = ax2.barh(
        [f"Sensor_{sid}" for sid in df_sorted['sensor_id']],
        df_sorted['shap_score'],
        color=bar_colors
    )
    ax2.set_xlabel('SHAP Score')
    ax2.set_title('Top Sensor SHAP Scores by Process')

    # 범례
    patches = [mpatches.Patch(color=COLORS.get(p, '#888'), label=p) for p in COLORS if p in processes]
    ax2.legend(handles=patches, loc='lower right')

    plt.tight_layout()
    plt.savefig('data/raw/process_contribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved: data/raw/process_contribution.png")

    # CSV 저장
    result_df = pd.DataFrame([
        {'process': p, 'contribution_pct': pct}
        for p, pct in sorted(process_pct.items(), key=lambda x: -x[1])
    ])
    result_df.to_csv('data/raw/process_contribution.csv', index=False)
    print("Saved: data/raw/process_contribution.csv")

    return process_pct, df

if __name__ == "__main__":
    analyze_process_contribution()
