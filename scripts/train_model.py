"""
train_model.py
เทรนโมเดล classify ท่ามือจาก data/gestures.csv แล้วบันทึกโมเดลไว้ที่ model.pkl

วิธีใช้:
    python train_model.py
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")


def make_classifier():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )


def count_label_runs(df):
    """นับจำนวน block ต่อเนื่องของ label เดียวกันในไฟล์ (เช่น A A A B B C C A -> 4 runs)"""
    runs = 0
    prev = None
    for label in df["label"]:
        if label != prev:
            runs += 1
            prev = label
    return runs


def is_temporally_ordered(df, max_runs_ratio=2.0):
    """
    เช็คว่าไฟล์ยังเรียงตามลำดับการเก็บจริงหรือถูก shuffle ไปแล้ว
    ใช้จำนวน 'block ต่อเนื่อง' ของแต่ละ label เทียบกับจำนวน label ทั้งหมด:
    เก็บแบบปกติ (เก็บทีละ label ต่อเนื่อง แม้จะย้อนกลับมาเก็บ label เดิมเพิ่มทีหลังบ้าง)
    จะได้ runs ใกล้เคียงจำนวน label เดิม ส่วนไฟล์ที่ถูก shuffle จะได้ runs พุ่งไปใกล้จำนวนแถวทั้งหมด
    """
    n_runs = count_label_runs(df)
    n_labels = df["label"].nunique()
    return n_runs, n_runs <= max_runs_ratio * n_labels


def temporal_split(df, test_frac=0.2):
    """แบ่ง 80% แรก / 20% ท้าย ตามลำดับการเก็บจริงในแต่ละ label"""
    train_parts, test_parts = [], []
    for _, group in df.groupby("label", sort=False):
        n_test = max(1, round(len(group) * test_frac))
        n_train = len(group) - n_test
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])
    return pd.concat(train_parts), pd.concat(test_parts)


def evaluate_split(X_train, y_train, X_test, y_test, labels, split_name):
    clf = make_classifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]

    print(f"\n=== {split_name} — {len(y_test)} test samples ===")
    print(f"Accuracy: {acc:.3f} | Macro F1: {macro_f1:.3f}")
    print(classification_report(y_test, y_pred, labels=labels))

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Confusion matrix (แถว = true label, คอลัมน์ = predicted label):")
    print(cm_df.to_string())

    return clf, acc, macro_f1


def main():
    if not os.path.exists(CSV_PATH):
        print(f"ไม่พบไฟล์ {CSV_PATH} — รัน collect_data.py ก่อน")
        return

    df = pd.read_csv("data/gestures.csv", header=None)
    
    # ถ้าบรรทัดแรกติดตัวหนังสือหัวตารางมา ให้ตัดทิ้งไป
    if str(df.iloc[0, 0]).strip() == 'label':
        df = df.iloc[1:].reset_index(drop=True)
        
    # บังคับตั้งชื่อคอลัมน์แรกให้เป็น 'label'
    df.rename(columns={0: "label"}, inplace=True)
    
    # แปลงข้อมูลพิกัดมือที่เหลือให้เป็นตัวเลขทั้งหมด ป้องกัน Error
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # ลบบรรทัดที่ข้อมูลพังหรือแหว่งทิ้ง
    df = df.dropna().reset_index(drop=True)
    print(f"จำนวนตัวอย่างทั้งหมด: {len(df)}")
    print("จำนวนตัวอย่างต่อ label:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        print("\nต้องมีอย่างน้อย 2 label ถึงจะเทรนได้ (แนะนำให้มี 'no_sign' ด้วย)")
        return

    labels = sorted(df["label"].unique())
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    # ---------- Random split (baseline เดิม) ----------
    min_class_count = df["label"].value_counts().min()
    stratify = y if min_class_count >= 2 else None
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    clf, acc_random, f1_random = evaluate_split(
        X_train_r, y_train_r, X_test_r, y_test_r, labels, "Random 80/20 split"
    )

    # ---------- Temporal split ----------
    n_runs, ordered = is_temporally_ordered(df)
    n_labels = df["label"].nunique()
    print(
        f"\nจำนวน block ต่อเนื่องของ label ในไฟล์: {n_runs} (label ทั้งหมด {n_labels}) "
        f"-> เรียงตามลำดับเก็บจริงหรือไม่: {ordered}"
    )

    if not ordered:
        print(
            "ไฟล์ gestures.csv น่าจะถูก shuffle ไปแล้ว (จำนวน block ต่อเนื่องของ label เยอะผิดปกติเทียบกับจำนวน label) "
            "ทำ temporal split แบบ 80% แรก/20% ท้ายต่อ label ไม่ได้อย่างมีความหมาย — ข้ามขั้นตอนนี้"
        )
        acc_temporal = f1_temporal = None
    else:
        train_df, test_df = temporal_split(df, test_frac=0.2)
        X_train_t = train_df.drop(columns=["label"]).values
        y_train_t = train_df["label"].values
        X_test_t = test_df.drop(columns=["label"]).values
        y_test_t = test_df["label"].values

        _, acc_temporal, f1_temporal = evaluate_split(
            X_train_t, y_train_t, X_test_t, y_test_t, labels,
            "Temporal split (80% แรก train / 20% ท้าย test ต่อ label ตามลำดับเก็บจริง)",
        )

    # ---------- สรุปเทียบ ----------
    print("\n=== สรุปเทียบ random split vs temporal split ===")
    print(f"{'Split':45} {'Accuracy':>10} {'Macro F1':>10}")
    print(f"{'Random 80/20':45} {acc_random:>10.3f} {f1_random:>10.3f}")
    if acc_temporal is not None:
        print(f"{'Temporal 80/20 (chronological per class)':45} {acc_temporal:>10.3f} {f1_temporal:>10.3f}")
    else:
        print(f"{'Temporal 80/20 (chronological per class)':45} {'N/A (csv shuffled)':>10}")

    # โมเดลที่บันทึกไว้ใช้งานจริง เทรนจาก random split เหมือนเดิม
    joblib.dump({"model": clf, "labels": labels}, MODEL_PATH)
    print(f"\nบันทึกโมเดลไว้ที่: {MODEL_PATH}")


if __name__ == "__main__":
    main()
