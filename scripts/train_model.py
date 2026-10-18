"""
train_model.py
เทรนโมเดล classify ท่ามือจาก data/gestures.csv แล้วบันทึกโมเดลไว้ที่ models/model.pkl
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
    runs = 0
    prev = None
    for label in df["label"]:
        if label != prev:
            runs += 1
            prev = label
    return runs


def is_temporally_ordered(df, max_runs_ratio=2.0):
    n_runs = count_label_runs(df)
    n_labels = df["label"].nunique()
    return n_runs, n_runs <= max_runs_ratio * n_labels


def temporal_split(df, test_frac=0.2):
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


def train_model_from_csv(csv_path=CSV_PATH, output_path=MODEL_PATH):
    """ฟังก์ชันหลักสำหรับสั่งเทรนโมเดล (เรียกใช้ได้ทั้งผ่านโค้ดและ main.py)"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ชุดข้อมูลที่: {csv_path}")

    df = pd.read_csv(csv_path, header=None)

    # ตัดหัวตารางถ้ามีคำว่า label
    if str(df.iloc[0, 0]).strip() == "label":
        df = df.iloc[1:].reset_index(drop=True)

    df.rename(columns={0: "label"}, inplace=True)

    # แปลงคอลัมน์ตัวเลขพิกัดมือ
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    print(f"จำนวนตัวอย่างทั้งหมด: {len(df)}")
    print("จำนวนตัวอย่างต่อ label:")
    print(df["label"].value_counts())

    if df["label"].nunique() < 2:
        raise ValueError("ต้องมีอย่างน้อย 2 label ถึงจะเทรนได้ (รวม 'no_sign')")

    labels = sorted(df["label"].unique())
    X = df.drop(columns=["label"]).values
    y = df["label"].values

    min_class_count = df["label"].value_counts().min()
    stratify = y if min_class_count >= 2 else None
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    clf, acc_random, f1_random = evaluate_split(
        X_train_r, y_train_r, X_test_r, y_test_r, labels, "Random 80/20 split"
    )

    # เทรนโมเดลตัวเต็มแล้วบันทึก
    final_clf = make_classifier()
    final_clf.fit(X, y)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump({"model": final_clf, "labels": labels}, output_path)
    print(f"\nบันทึกโมเดลไว้ที่: {output_path}")
    return labels


def main():
    train_model_from_csv()


if __name__ == "__main__":
    main()