"""
Comprehensive Benchmark Execution Engine for Quantum Kernel Methods on MedMNIST.
Orchestrates preprocessing, quantum state embedding, kernel construction,
QSVM training, deep metric extraction, classical baselines, parametric sweeps,
and publication figure generation.
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import time
import numpy as np

from src.config import (
    DATASETS_CONFIG,
    PATCH_CONFIGS,
    DEFAULT_ALPHA,
    ALPHA_SWEEP_VALUES,
    GRAM_DIR,
    LOGS_DIR,
    FIGURES_DIR,
)
from src.data.loader import load_medmnist_dataset
from src.data.preprocessor import QuantumPreprocessor
from src.quantum.circuits import (
    generate_angle_statevectors_vectorized,
    generate_entangled_statevectors_vectorized,
)
from src.quantum.kernels import (
    compute_global_kernel,
    compute_local_kernel,
    compute_multiscale_kernel,
    compute_hierarchical_multiscale_kernel,
)
from src.models.qsvm import QSVMClassifier
from src.models.baselines import train_classical_svm_baseline
from src.metrics.spectral import compute_effective_rank
from src.metrics.alignment import compute_centered_kta, compute_target_matrix
from src.metrics.concentration import compute_concentration_statistics
from src.metrics.classification import evaluate_classification_metrics
from src.visualization.plots import (
    plot_gram_matrix_heatmaps,
    plot_eigenvalue_spectra,
    plot_concentration_distributions,
    plot_multiscale_alpha_sweep,
)


def run_benchmark_for_dataset(
    dataset_key: str,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    seed: int = 42,
    tune_C: bool = True,
    run_alpha_sweep: bool = True,
    save_artifacts: bool = True,
) -> Dict[str, Any]:
    """
    Run full benchmark pipeline on a single MedMNIST dataset.
    """
    start_time = time.time()
    config = DATASETS_CONFIG[dataset_key]
    n_classes = config["n_classes"]
    d_name = config["name"]
    print(f"\n=======================================================")
    print(f"BENCHMARKING: {d_name} ({dataset_key}) | Task: {config['task']}")
    print(f"Classes: {n_classes} | Modality: {config['description']}")
    print(f"=======================================================")

    # 1. Load Data
    print("[1/6] Loading and preprocessing dataset...")
    X_train, y_train, X_test, y_test = load_medmnist_dataset(
        dataset_key=dataset_key,
        n_train=n_train,
        n_test=n_test,
        seed=seed,
    )
    print(f"      Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"      Class distribution (train): {np.bincount(y_train)}")

    # 2. Preprocessing (PCA strictly fitted on train + MinMaxScaler to [-pi, pi])
    print("[2/6] Fitting PCA (8 components) and MinMaxScaler to [-pi, pi] on Train...")
    preprocessor = QuantumPreprocessor(n_components=8)
    angles_train = preprocessor.fit_transform(X_train)
    angles_test = preprocessor.transform(X_test)
    pca_summary = preprocessor.get_summary()
    print(f"      PCA Explained Variance: {pca_summary['total_explained_variance']:.2%}")

    # 3. Quantum State Generation
    print("[3/6] Generating 8-qubit quantum statevectors (|psi> in C^256)...")
    # Pure Angle Embedding
    states_train = generate_angle_statevectors_vectorized(angles_train)
    states_test = generate_angle_statevectors_vectorized(angles_test)

    # Entangled Feature Map
    states_train_ent = generate_entangled_statevectors_vectorized(angles_train)
    states_test_ent = generate_entangled_statevectors_vectorized(angles_test)

    # 4. Kernel Constructions (Gram matrices)
    print("[4/6] Constructing Global, Local (1, 2, 4-qubit), and Multiscale Gram matrices...")
    gram_train: Dict[str, np.ndarray] = {}
    gram_test: Dict[str, np.ndarray] = {}

    # Global Fidelity
    gram_train["global"] = compute_global_kernel(states_train)
    gram_test["global"] = compute_global_kernel(states_train, states_test)

    # Local Subsystem Kernels
    for p_type in ["1-qubit", "2-qubit", "4-qubit"]:
        key_name = f"local_{p_type.replace('-', '')}"
        gram_train[key_name] = compute_local_kernel(states_train, patch_type=p_type)
        gram_test[key_name] = compute_local_kernel(states_train, states_test, patch_type=p_type)

    # Multiscale (alpha=0.5 default)
    gram_train["multiscale"] = compute_multiscale_kernel(
        gram_train["global"], gram_train["local_2qubit"], alpha=DEFAULT_ALPHA
    )
    gram_test["multiscale"] = compute_multiscale_kernel(
        gram_test["global"], gram_test["local_2qubit"], alpha=DEFAULT_ALPHA
    )

    # Hierarchical Multiscale
    gram_train["hierarchical"] = compute_hierarchical_multiscale_kernel(
        gram_train["local_1qubit"],
        gram_train["local_2qubit"],
        gram_train["local_4qubit"],
        gram_train["global"],
    )
    gram_test["hierarchical"] = compute_hierarchical_multiscale_kernel(
        gram_test["local_1qubit"],
        gram_test["local_2qubit"],
        gram_test["local_4qubit"],
        gram_test["global"],
    )

    # Entangled Variants
    gram_train["global_entangled"] = compute_global_kernel(states_train_ent)
    gram_test["global_entangled"] = compute_global_kernel(states_train_ent, states_test_ent)
    gram_train["local_2qubit_entangled"] = compute_local_kernel(states_train_ent, patch_type="2-qubit")
    gram_test["local_2qubit_entangled"] = compute_local_kernel(states_train_ent, states_test_ent, patch_type="2-qubit")
    gram_train["multiscale_entangled"] = compute_multiscale_kernel(
        gram_train["global_entangled"], gram_train["local_2qubit_entangled"], alpha=DEFAULT_ALPHA
    )
    gram_test["multiscale_entangled"] = compute_multiscale_kernel(
        gram_test["global_entangled"], gram_test["local_2qubit_entangled"], alpha=DEFAULT_ALPHA
    )

    # 5. Metric Extraction and QSVM Evaluation
    print("[5/6] Extracting deep metrics and training QSVM classifiers...")
    kernel_results: Dict[str, Any] = {}
    spectral_profiles: Dict[str, Any] = {}

    target_matrix = compute_target_matrix(y_train, n_classes=n_classes)

    for k_name, K_tr in gram_train.items():
        K_te = gram_test[k_name]

        # Spectral analysis
        spectral = compute_effective_rank(K_tr)
        spectral_profiles[k_name] = spectral

        # KTA
        kta = compute_centered_kta(K_tr, y_train, n_classes=n_classes)

        # Off-diagonal statistics
        conc_stats = compute_concentration_statistics(K_tr)

        # Train QSVM
        qsvm = QSVMClassifier(random_state=seed)
        qsvm.fit(K_tr, y_train, tune_C=tune_C)
        preds = qsvm.predict(K_te)
        cls_metrics = evaluate_classification_metrics(y_test, preds, n_classes=n_classes)

        kernel_results[k_name] = {
            "effective_rank": spectral["effective_rank"],
            "effective_rank_ratio": spectral["effective_rank_ratio"],
            "spectral_entropy": spectral["entropy"],
            "kta": kta,
            "offdiag_mean": conc_stats["offdiag_mean"],
            "offdiag_var": conc_stats["offdiag_var"],
            "offdiag_std": conc_stats["offdiag_std"],
            "concentration_index": conc_stats["concentration_index"],
            "best_C": qsvm.best_C,
            "accuracy": cls_metrics["accuracy"],
            "balanced_accuracy": cls_metrics["balanced_accuracy"],
            "macro_f1": cls_metrics["macro_f1"],
            "weighted_f1": cls_metrics["weighted_f1"],
            "confusion_matrix": cls_metrics["confusion_matrix"],
        }
        print(f"      Kernel: {k_name:<22} | BalAcc: {cls_metrics['balanced_accuracy']:.2%} | Acc: {cls_metrics['accuracy']:.2%} | KTA: {kta:.4f} | EffRank: {spectral['effective_rank']:.1f} | Var: {conc_stats['offdiag_var']:.2e}")

    # Classical baselines on the 8 PCA features
    print("      Evaluating Classical Baselines (RBF and Linear SVM on 8 PCA features)...")
    for baseline_kern in ["rbf", "linear"]:
        base_res = train_classical_svm_baseline(angles_train, y_train, angles_test, kernel=baseline_kern, tune_C=tune_C)
        b_metrics = evaluate_classification_metrics(y_test, base_res["test_preds"], n_classes=n_classes)
        kernel_results[f"classical_{baseline_kern}"] = {
            "effective_rank": None,
            "effective_rank_ratio": None,
            "kta": None,
            "offdiag_mean": None,
            "offdiag_var": None,
            "accuracy": b_metrics["accuracy"],
            "balanced_accuracy": b_metrics["balanced_accuracy"],
            "macro_f1": b_metrics["macro_f1"],
            "weighted_f1": b_metrics["weighted_f1"],
            "best_C": base_res["best_C"],
            "confusion_matrix": b_metrics["confusion_matrix"],
        }
        print(f"      Baseline: classical_{baseline_kern:<12} | BalAcc: {b_metrics['balanced_accuracy']:.2%} | Acc: {b_metrics['accuracy']:.2%}")

    # 6. Parametric Alpha Sweep (Multiscale)
    alpha_sweep_data = {}
    if run_alpha_sweep:
        print("[6/6] Executing Multiscale Alpha Sweep (alpha in [0.0, 1.0])...")
        sweep_alphas = ALPHA_SWEEP_VALUES
        sweep_bal_acc = []
        sweep_acc = []
        sweep_kta = []
        sweep_eff_rank = []
        sweep_var = []

        for a in sweep_alphas:
            K_multi_tr_a = compute_multiscale_kernel(gram_train["global"], gram_train["local_2qubit"], alpha=a)
            K_multi_te_a = compute_multiscale_kernel(gram_test["global"], gram_test["local_2qubit"], alpha=a)

            # Evaluate
            eff_r = compute_effective_rank(K_multi_tr_a)["effective_rank"]
            kta_val = compute_centered_kta(K_multi_tr_a, y_train, n_classes=n_classes)
            var_val = compute_concentration_statistics(K_multi_tr_a)["offdiag_var"]

            qsvm_a = QSVMClassifier(random_state=seed)
            qsvm_a.fit(K_multi_tr_a, y_train, tune_C=tune_C)
            preds_a = qsvm_a.predict(K_multi_te_a)
            m_a = evaluate_classification_metrics(y_test, preds_a, n_classes=n_classes)

            sweep_bal_acc.append(m_a["balanced_accuracy"])
            sweep_acc.append(m_a["accuracy"])
            sweep_kta.append(kta_val)
            sweep_eff_rank.append(eff_r)
            sweep_var.append(var_val)

        alpha_sweep_data = {
            "alphas": sweep_alphas,
            "balanced_accuracy": sweep_bal_acc,
            "accuracy": sweep_acc,
            "kta": sweep_kta,
            "effective_rank": sweep_eff_rank,
            "offdiag_var": sweep_var,
        }

    # Visualizations
    if save_artifacts:
        print("      Generating publication figures...")
        # 1. Heatmaps
        plot_gram_matrix_heatmaps(
            gram_dict=gram_train,
            target_matrix=target_matrix,
            dataset_name=d_name,
            save_path=FIGURES_DIR / f"heatmaps/{dataset_key}_heatmaps.png",
        )
        # 2. Spectra
        plot_eigenvalue_spectra(
            spectral_dict=spectral_profiles,
            dataset_name=d_name,
            save_path=FIGURES_DIR / f"spectra/{dataset_key}_spectra.png",
        )
        # 3. Concentration Distributions
        plot_concentration_distributions(
            gram_dict=gram_train,
            dataset_name=d_name,
            save_path=FIGURES_DIR / f"concentration/{dataset_key}_concentration.png",
        )
        # 4. Alpha Sweep
        if run_alpha_sweep:
            plot_multiscale_alpha_sweep(
                alphas=sweep_alphas,
                sweep_metrics=alpha_sweep_data,
                dataset_name=d_name,
                save_path=FIGURES_DIR / f"multiscale_sweep/{dataset_key}_alpha_sweep.png",
            )

        # Save Gram matrices
        np.savez_compressed(
            GRAM_DIR / f"{dataset_key}_gram_matrices.npz",
            K_global_train=gram_train["global"],
            K_global_test=gram_test["global"],
            K_local_2q_train=gram_train["local_2qubit"],
            K_local_2q_test=gram_test["local_2qubit"],
            K_multiscale_train=gram_train["multiscale"],
            K_multiscale_test=gram_test["multiscale"],
            y_train=y_train,
            y_test=y_test,
        )

        # Save summary JSON log
        summary_log = {
            "dataset": dataset_key,
            "dataset_name": d_name,
            "n_classes": n_classes,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pca_summary": pca_summary,
            "kernel_results": kernel_results,
            "alpha_sweep": alpha_sweep_data,
            "elapsed_seconds": time.time() - start_time,
        }
        with open(LOGS_DIR / f"{dataset_key}_benchmark_results.json", "w") as f:
            json.dump(summary_log, f, indent=2)

    total_time = time.time() - start_time
    print(f"Finished {d_name} in {total_time:.2f}s.\n")
    return {
        "dataset": dataset_key,
        "dataset_name": d_name,
        "kernel_results": kernel_results,
        "alpha_sweep": alpha_sweep_data,
        "elapsed_seconds": total_time,
    }
