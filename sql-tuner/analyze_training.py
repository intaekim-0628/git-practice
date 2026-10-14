"""Render real epoch records only. Never manufacture unrun epoch metrics."""
import argparse,json
from pathlib import Path

def render(output_dir):
    root=Path(output_dir);path=root/'epoch_analysis.json'
    if not path.exists():raise ValueError('epoch_analysis.json 없음: 학습이 실행되지 않았거나 첫 평가가 완료되지 않았습니다.')
    reports=json.loads(path.read_text());lines=['# 실제 epoch 분석','', '| Epoch | Train loss¹ | Eval loss | Eval 변화 | Perplexity | 학습·평가 초 | 과적합 신호 |','|---|---:|---:|---:|---:|---:|---|']
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    for r in reports:
        lines.append(f"| {r['epoch']:.3f} | {fmt(r.get('train_loss'))} | {fmt(r['eval_loss'])} | {fmt(r['eval_loss_delta'])} | {fmt(r['perplexity'])} | {r['epoch_train_eval_seconds']:.1f} | {'있음' if r['possible_overfit'] else '검출 안 됨'} |")
        if not r['completed_epoch']:lines.append(f"\nEpoch {r['epoch']:.3f}는 시간 제한으로 중단된 부분 epoch입니다.\n")
    lines += ['', '¹ Train loss는 해당 epoch의 optimizer-step 로그 평균이며 샘플/토큰 가중 평균이 아닙니다. Eval loss도 Trainer의 배치 평균 집계입니다. 서로 직접 비교할 때 길이 차이를 고려하세요.', '', 'Eval loss 감소는 참조 정답 예측 향상입니다. SQL 결과 동등성이나 실제 Oracle 성능 개선의 증거가 아닙니다. Perplexity는 동일한 답변 토큰 마스킹 기준의 exp(eval_loss)입니다.']
    baseline=root/'baseline_metrics.json'
    if baseline.exists():
        val=json.loads(baseline.read_text()).get('baseline_loss')
        if val is not None:lines.append(f'\n학습 전 eval loss: {val:.4f}; 마지막 epoch 변화: {reports[-1]["eval_loss"]-val:+.4f}. 마지막 epoch와 선택된 best checkpoint는 다를 수 있습니다.')
    for r in reports:
        if r.get('generation_metrics'):lines.append(f"\nEpoch {r['epoch']:.3f} 생성 검사(고정된 소수 eval 표본): `{json.dumps(r['generation_metrics'],ensure_ascii=False)}`. 구조·규칙 준수 검사이며 DB 실행 검증은 아닙니다.")
    plan=root/'time_plan.json'
    if plan.exists():
        p=json.loads(plan.read_text());lines.append(f"\n첫 epoch 측정 기반 선택: {p['selected_epochs']} epochs, 예상 학습 {p['estimated_training_hours']:.2f}시간(체크포인트·최종 test 제외).")
        if p['under_three_hours_at_cap']:lines.append('데이터가 작아 설정된 최대 epoch로 3시간을 채울 수 없습니다. 시간만 채우기 위한 반복 대신 전문가 검증을 거친 새 쿼리/실행 계획을 추가하세요.')
    summary=root/'summary.json'
    if summary.exists():lines.append('\n최종 결과: `'+summary.read_text().strip()+'`')
    (root/'epoch_analysis.md').write_text('\n'.join(lines)+'\n')
    return root/'epoch_analysis.md'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output_dir');print(render(p.parse_args().output_dir))
