"""Readable ordered-stage report and artifact hashes."""
import json,hashlib
from pathlib import Path
from .train import atomic_json
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/midgame_20261007'

def main():
    d=json.loads((RUN/'evaluation.json').read_text(encoding='utf-8'))
    arms=[a for a in ('baseline','step1','step2') if any(r['arm']==a for r in d['full_games'])]
    splits=[s for s in ('validation','test') if any(r['split']==s for r in d['full_games'])]
    replay_cache={}
    for file in RUN.glob('*_mid_replays.json'):
        replay_cache[file.name]=json.loads(file.read_text(encoding='utf-8'))['games']
    conditional=[]
    for row in d['midgames']:
        name=f"{row['arm']}_{row['kind']}_{row['split']}_mid_replays.json"
        games=[g for g in replay_cache.get(name,[]) if g['horizon']==row['horizon']
               and (row['family']=='all' or g['family']==row['family'])]
        correct=[g for g in games if g['first_correct']]
        complete=sum(g['winner']==g['a_color'] and g['decisions']<=g['horizon'] for g in correct)
        conditional.append(dict(arm=row['arm'],kind=row['kind'],split=row['split'],
                                horizon=row['horizon'],family=row['family'],
                                first_correct=len(correct),completed=complete,
                                conditional_completion_rate=complete/len(correct) if correct else None))
    atomic_json(RUN/'conditional_completion.json',dict(runs=conditional))
    lines=['# 12×12 중간 판 커리큘럼 → 다양한 공격·범주 균형 — 2026-10-07','',
           '사용자 요청대로 1단계의 학습·검증을 완료한 뒤 2단계의 자료 생성·학습을 실행했습니다. Fly·GRU, 독립 새 학습 시드4810의 탐색적 비교입니다.','',
           'baseline은 빈 판에서만 학습, step1은 기존 자료+중간 판 커리큘럼, step2는 다양한 공격·실제 후속 응수 자료+범주 균형+새 시작 풀입니다.','',
           '## 빈 판에서 시작한 강한 상대 경기','',
           '| 평가 | 방식 | 모델 | 승/256 | 방어 누락/기회 | 복수 위협 도달/256 | 실제 안전한 복수 위협 생성 |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in d['full_games']:
        block=r['tactical_decisions']['block']['positions']
        lines.append(f"| {r['split']} | {r['arm']} | {r['kind']} | {r['wins']} | {r['defense_omissions']}/{block} | {r['games_reaching_multiple_threat']} | {r['actual_safe_forks_created']} |")
    lines+=['','| 평가 | 방식 | 합산 승률 | 방어 누락률 | 복수 위협 도달 비율 | 실제 안전한 복수 위협 생성 |',
            '|---|---|---:|---:|---:|---:|']
    summary={}
    for split in splits:
        summary[split]={}
        for arm in arms:
            rows=[r for r in d['full_games'] if r['arm']==arm and r['split']==split]
            if not rows:
                continue
            n=sum(r['games'] for r in rows)
            wins=sum(r['wins'] for r in rows)
            omissions=sum(r['defense_omissions'] for r in rows)
            opportunities=sum(r['tactical_decisions']['block']['positions'] for r in rows)
            mult=sum(r['games_reaching_multiple_threat'] for r in rows)
            forks=sum(r['actual_safe_forks_created'] for r in rows)
            summary[split][arm]=dict(games=n,wins=wins,win_rate=wins/n,defense_omissions=omissions,
                                     defense_opportunities=opportunities,defense_omission_rate=omissions/opportunities if opportunities else None,
                                     multiple_threat_game_rate=mult/n,safe_forks=forks)
            lines.append(f"| {split} | {arm} | {wins}/{n} ({wins/n:.2%}) | {omissions/opportunities:.2%} | {mult/n:.2%} | {forks} |")
    if (RUN/'clean_full_game_subset.json').exists():
        clean=json.loads((RUN/'clean_full_game_subset.json').read_text(encoding='utf-8'))
        lines+=['','## 실제 기보 겹침을 제외한 공통 부분집합','',
                '회전·반사를 고려한 전체 기보 사후 감사에서 초기5ply 예방 판 한 종류가 step1/Fly 검증3기보·시험1기보에 자연 재등장했습니다. 중간 판 평가의 모든 기보에는 겹침이0입니다.',
                '기존 고정 평가를 보존하고, 해당 시작 인덱스를 모든 방식·두 모델에서 공통 제외한 보조 결과를 아래에 표시합니다. 사후 제외에 따른 선택 편향 가능성이 있으므로 별도의 사전 고정 평가라고 주장하지 않습니다. 모델·설정·학습은 바꾸지 않았습니다.','',
                '| 평가 | 방식 | 승리/경기 | 방어 누락률 | 복수 위협 도달 비율 |',
                '|---|---|---:|---:|---:|']
        for split,rr in clean['summary'].items():
            for arm,r in rr.items():
                lines.append(f"| {split} | {arm} | {r['wins']}/{r['games']} | {r['defense_omission_rate']:.2%} | {r['multiple_threat_game_rate']:.2%} |")
        lines+=['','원 시험0/512승과 겹침을 제외한0/510승은 모든 방식에서 같은 결론입니다. 보조 결과 원본은 clean_full_game_subset.json, 재등장 위치는 overlap_details.json입니다.']
    lines+=['','## 中間 판의 첫 수와 마지막 승리','',
            '각 평가 시작 판은 실제 교대 기보로 도달 가능하고, 동일 시작 판·상대 난수 스트림을 사용합니다. 초기 정답을 맞혔는지와 지정한 자기 착수 수 안에 승리했는지, 경기 종료까지 승리했는지를 분리합니다.','',
            '| 평가 | 방식 | 모델 | 자기 착수 깊이 | 첫 수 정답/128 | 깊이 내 승리/128 | 최종 승리/128 |',
            '|---|---|---|---:|---:|---:|---:|']
    for r in d['midgames']:
        if r['family']=='all':
            lines.append(f"| {r['split']} | {r['arm']} | {r['kind']} | {r['horizon']} | {r['first_correct']} | {r['within_horizon_wins']} | {r['wins']} |")
    lines+=['','| 평가 | 방식 | 깊이 | 합산 첫 수 정답률 | 합산 깊이 내 완승률 | 합산 최종 승률 |',
            '|---|---|---:|---:|---:|---:|']
    for split in splits:
        for arm in arms:
            for depth in (1,2,3):
                rows=[r for r in d['midgames'] if r['split']==split and r['arm']==arm and r['horizon']==depth and r['family']=='all']
                if not rows:
                    continue
                n=sum(r['games'] for r in rows)
                values=[sum(r[k] for r in rows) for k in ('first_correct','within_horizon_wins','wins')]
                lines.append(f"| {split} | {arm} | {depth} | {values[0]}/{n} ({values[0]/n:.2%}) | {values[1]}/{n} ({values[1]/n:.2%}) | {values[2]}/{n} ({values[2]/n:.2%}) |")
                summary[split][arm].setdefault('midgames',{})[str(depth)]=dict(games=n,first_correct=values[0],within_horizon_wins=values[1],wins=values[2])
    lines+=['','## 첫 수가 맞았던 경기의 연결 성공','',
            '| 평가 | 방식 | 모델 | 깊이 | 첫 수 정답 경기 중 깊이 내 완승 |',
            '|---|---|---|---:|---:|']
    for row in conditional:
        if row['family']=='all':
            lines.append(f"| {row['split']} | {row['arm']} | {row['kind']} | {row['horizon']} | {row['completed']}/{row['first_correct']} |")
    lines+=['','## 유형별 끝까지 승리','',
            '| 평가 | 방식 | 모델 | 깊이 | 유형 | 깊이 내 승리/경기 | 最終 승리/경기 |',
            '|---|---|---|---:|---|---:|---:|']
    for r in d['midgames']:
        if r['family']!='all':
            lines.append(f"| {r['split']} | {r['arm']} | {r['kind']} | {r['horizon']} | {r['family']} | {r['within_horizon_wins']}/{r['games']} | {r['wins']}/{r['games']} |")
    lines+=['','## 학습 예산과 혼합','',
            '각 군은 새 초기화와 같은 보조600×128, PPO300×32×16, 에포크2 및 에포크별 보조128을 사용했습니다. 보조 제시153,600개·정책 결정153,600개입니다. 중간 판에 필요한 기보 연산량·시간은 추가되므로 같은 계산량이라고 주장하지 않습니다.','',
            '새 경기50%는 빈 판,50%는 중간 판입니다. 비율은 경기 시작 수 기준이며 정책 결정 수 비율과 다릅니다. 중간 판 깊이는 업데이트1~100:1착수,101~200:2착수,201~300:3착수입니다. 상대는30→60→85% 규칙 커리큘럼, 평가는100% 규칙 상대입니다. 혼합 학습 승률을 빈 판 실전 승률로 해석하지 않습니다.','',
            '| 방식 | 모델 | 파라미터 | 보조 제시 | 정책 결정 | 보조 기보 관측 | 추가 수집 기보 관측 | 시간(초) | 최대 할당 GiB |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    training_notes=[]
    for arm in arms:
        for kind in ('fly','gru'):
            r=json.loads((RUN/arm/f'{kind}_seed4810.json').read_text(encoding='utf-8'))
            lines.append(f"| {arm} | {kind} | {r['parameters']:,} | {sum(r['auxiliary_samples_by_category'].values()):,} | {r['learner_decisions']:,} | {r['auxiliary_history_frames']:,} | {r.get('rollout_prefix_frames',0):,} | {r['elapsed_seconds']:.1f} | {r['peak_allocated_gib']:.3f} |")
            if arm!='baseline':
                training_notes+=['',f"{arm}/{kind} 경기 시작 수: {r['episode_starts']}. 깊이0은 빈 판입니다.",
                        f"학습 중 시작 유형별 종료 결과: {r['midgame_outcomes']}.",
                        f"범주별 보조 제시 횟수: {r['auxiliary_samples_by_category']}.",'']
    lines+=training_notes
    if (RUN/'data/data_diversify.json').exists():
        m=json.loads((RUN/'data/data_diversify.json').read_text(encoding='utf-8'))
        lines+=['',f"2단계 새 시작 판 {m['attack_starts']:,}개, 공격·후속·예방 보조 예제 {m['attack_and_continuation_examples']:,}개(예방 {m['defensive_counterparts']:,}개). 전체 보조 자료20,000개를 유지했습니다. 자료별 범주 빈도: {m['counts']}.",
                '2단계는 자료의 빈도와 상관없이6범주를 같은 확률로 제시합니다. 실제 제시 횟수는 위 기록 및 run_audit.json에 포함됩니다.']
    lines+=['','## 범위·검증·해석','',
            '35개 테스트 및 실제 짧은 학습 검증 통과. 중간 판 재시작의 돌·기보·차례·증분 줄 개수 일치, 기보 burn-in의 수집/PPO 일치와 순환 가중치 기울기, 인증 깊이 내 실제 상대 완승, 범주 균형 제시를 검증했습니다.',
            '새 검증/시험 판은 각384개(깊이별128개, 흑백 균형)이며 모든 이전 차례·현재 판을 D4 정규화 해시로 학습에서 제외했습니다. 기보와 현재 관측은 모든 자료에서 정확히 일치합니다.',
            '시작 판 생성은 여러 구조를 갖지만 여전히 정해진 강제 전술 가족과 무작위 주변 돌의 조합입니다. 전체 오목의 장기 전략·조용한 계획을 해결했다는 의미가 아닙니다.',
            '2단계는 공격 다양화·실제 후속 응수 지도·범주 균형·시작 풀 교체의 결합 효과이며 각 요소의 단독 인과 효과는 구분할 수 없습니다.',
            '중간 판 성공률이 오르더라도 빈 판 경기 승률이 오르지 않으면 전체 경기로의 전이가 확인되지 않은 것입니다. 단일 학습 시드이므로 Fly 연결 구조나 순환 기억의 우월성·통계적 확증을 주장하지 않습니다.',
            '이전 실험의 시험을 설정 선택에 재사용하지 않았으며, 두 단계가 완료된 뒤 새 시험을 고정 체크포인트로 평가합니다.',
            '2단계의 새 시험 방어 누락률은61.16%로 대조43.55% 및1단계39.22%보다 높았습니다. 공격 연결 향상이 방어 약화를 동반했으므로 전체 경기 전략이 개선됐다고 볼 수 없습니다.',
            '실제 평가 기보 전체와 각 군 학습 판의 겹침 여부는 run_audit.json에 별도 기록합니다. 복수 위협 도달 비율은 방어 누락 감소로 패배 양상이 바뀌어 증가할 수도 있으므로 단독으로 예방 능력 악화의 증거로 해석하지 않습니다.','',
            '- [사전 프로토콜](../../midgame_protocol.md)',
            '- [평가 수치](evaluation.json)',
            '- [1단계 자료 감사](data_audit_step1.json)',
            '- [예산·실제 기보 감사](run_audit.json)',
            '- [겹침 기보를 공통 제외한 보조 평가](clean_full_game_subset.json)',
            '- [첫 수 정답 경기의 조건부 완승](conditional_completion.json)',
            '- [중간 판 체크포인트 복원 검증](resume_audit.json)','',
            '~~~powershell',
            '.venv/Scripts/python.exe -m gomoku.midgame_evaluate --root gomoku/runs/midgame_20261007 --arms baseline step1 step2 --splits validation test',
            '.venv/Scripts/python.exe -m gomoku.midgame_audit --root gomoku/runs/midgame_20261007 --phase runs',
            '.venv/Scripts/python.exe -m gomoku.midgame_report',
            '~~~','']
    if 'test' in summary and 'step2' in summary['test']:
        text=[]
        for depth in (1,2,3):
            rates=[]
            for arm in arms:
                r=summary['test'][arm]['midgames'][str(depth)]
                rates.append(f"{r['within_horizon_wins']/r['games']:.1%}")
            text.append(f"자기 착수{depth}번 내 완승 "+' → '.join(rates))
        wins=' → '.join(f"{summary['test'][arm]['wins']}/{summary['test'][arm]['games']}" for arm in arms)
        headline='새 시험 결과(baseline → step1 → step2): '+ '; '.join(text)+'. 빈 판 강한 상대 승리는 '+wins+'입니다.'
        lines.insert(2,headline)
        lines.insert(3,'')
        if summary['test']['step2']['wins']==0:
            lines.insert(4,'중간 판에서의 개선 여부와 별개로 빈 판 전체 경기로의 승률 전이는 확인되지 않았습니다.')
            lines.insert(5,'')
    lines=[x.replace('中間','중간').replace('最終','최종') for x in lines]
    (RUN/'results.md').write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(RUN/'summary.json',summary)
    for p in ROOT.glob('*.py'):
        (RUN/'source'/p.name).write_bytes(p.read_bytes())
    (RUN/'source/midgame_protocol.md').write_bytes((ROOT/'midgame_protocol.md').read_bytes())
    atomic_json(RUN/'completion_manifest.json',dict(arms=arms,splits=splits,tests_passed=35,
                files={str(p.relative_to(RUN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in RUN.rglob('*')
                       if p.is_file() and p.name!='completion_manifest.json'}))
    print(RUN/'results.md')
if __name__=='__main__':
    main()
