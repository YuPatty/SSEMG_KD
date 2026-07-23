"""
check_loss_balance.py
快速檢查 distill CSV log 裡各 loss 分量的加權後貢獻比例，確認沒有
某一項的梯度貢獻被另一項蓋過（response 四項之間、以及 relation vs 其他項）。

用法：
    python check_loss_balance.py distill_crossarch_log.csv \
        --w_mask 3.5 --w_mag 5.0 --w_pha 0.3 --w_com 2.3 \
        --w_feature 0.3 --w_relation 20.0
"""
import argparse
import csv

def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv_path')
    p.add_argument('--w_mask', type=float, default=3.5)
    p.add_argument('--w_mag', type=float, default=5.0)
    p.add_argument('--w_pha', type=float, default=0.3)
    p.add_argument('--w_com', type=float, default=2.3)
    p.add_argument('--w_feature', type=float, default=0.3)
    p.add_argument('--w_relation', type=float, default=20.0)
    args = p.parse_args()

    with open(args.csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    print(f"{'epoch':>5} | {'mask%':>6} {'mag%':>6} {'pha%':>6} {'com%':>6} "
          f"{'feat%':>6} {'rel%':>6} | total_weighted")
    for r in rows:
        vals = {
            'mask': float(r['resp_mask']) * args.w_mask,
            'mag':  float(r['resp_mag'])  * args.w_mag,
            'pha':  float(r['resp_pha'])  * args.w_pha,
            'com':  float(r['resp_com'])  * args.w_com,
            'feat': float(r['feat_loss']) * args.w_feature,
            'rel':  float(r.get('rel_loss') or 0) * args.w_relation,
        }
        total = sum(vals.values())
        if total == 0:
            continue
        pct = {k: v / total * 100 for k, v in vals.items()}
        print(f"{r['epoch']:>5} | {pct['mask']:6.1f} {pct['mag']:6.1f} {pct['pha']:6.1f} "
              f"{pct['com']:6.1f} {pct['feat']:6.1f} {pct['rel']:6.1f} | {total:.5f}")

if __name__ == '__main__':
    main()
