# Quantum-6-Single_Qubit_Gates 



import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)

FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

TITLE = "Quantum Single Qubit Gates"

N = 39
K = 7
QUBITS = 6
DIM = 64

WINDOWS = (1, 3)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

BATCH_SIZE = 256
MAX_ITERATIONS = 150
GRADIENT_TOLERANCE = 1e-5

BITS = (
    np.arange(DIM)[:, None] >> np.arange(QUBITS)[None, :]
) & 1

BIT_SIGNS = (2.0 * BITS.T - 1.0)


# OSNOVNA IDEJA
#
# Svaki kubit pocinje u |0>.
# Primenjuju se RY(a), RZ(b), RY(c).
# Relativna faza b utice na merljive verovatnoce
# kroz poslednju RY rotaciju.
#
# Uglovi a i b zavise od prethodnih kola.
# Parametri tih uglova i ugao c uce se zajedno.
#
# Svaki registar ostaje proizvodno stanje sest kubita.
# Sedam registara ima zasebne naucene parametre.
#
# OBUKA
#
# Regresija Bornove raspodele preko Brierovog gubitka.
# Klasicni optimizator menja parametre kvantnih kapija.
# Nema zasebnog klasicnog prediktivnog regresora.
#
# Hronoloska validacija bira prozor istorije i regularizaciju.
# Test ostaje odvojen od izbora hiperparametara.
# Zavrsna obuka koristi sve raspolozive istorijske prelaze.
#
# NEXT
#
# Logaritmi se racunaju direktno iz amplituda.
# Globalna normalizacija obuhvata sve legalne kombinacije.
# Dinamicko programiranje daje globalni maksimum raspodele.
#
# SIMULACIJA
#
# Algebraicki simulator izvodi ista jednokubitna kola kao Qiskit.
# Statevector nezavisno proverava njihove amplitude.
# Nema uzorkovanja, merenja ili slucajnog generisanja kombinacija.



class DistributionError(ArithmeticError):
    pass


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    reader = csv.reader(
        raw.decode("utf-8-sig").splitlines()
    )

    for line_number, row in enumerate(reader, start=1):
        try:
            values = tuple(int(value.strip()) for value in row)
        except ValueError as error:
            raise ValueError(
                f"{path.name}: necelobrojan red {line_number}."
            ) from error

        check(
            len(values) == K
            and len(set(values)) == K
            and min(values) >= 1
            and max(values) <= N
            and values == tuple(sorted(values)),
            f"{path.name}: nelegalan red {line_number}.",
        )
        rows.append(values)

    check(
        len(rows) >= 5,
        f"{path.name}: nedovoljno kola za trening, validaciju i test.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def history_features(data, times, window):
    times = np.asarray(times, dtype=np.int64)

    check(
        window in WINDOWS
        and times.ndim == 1
        and len(times) > 0
        and np.all((times >= 1) & (times <= len(data))),
        "Neispravan prozor ili vreme istorijskog ulaza.",
    )

    features = np.zeros(
        (len(times), 1 + 8 * window),
        dtype=float,
    )
    features[:, 0] = 1.0
    scale = np.sqrt(float(window))

    for lag in range(1, window + 1):
        source = times - lag
        available = source >= 0
        start = 1 + 8 * (lag - 1)

        features[available, start:start + K] = (
            data[source[available]].astype(float)
            + 1.0
            - 20.0
        ) / (19.0 * scale)

        # Oznaka dostupnosti razlikuje nedostajuci deo istorije.
        features[available, start + K] = 1.0 / scale

    return features


def initial_parameters(feature_count):
    indices = np.arange(
        QUBITS * feature_count,
        dtype=float,
    ).reshape(QUBITS, feature_count)

    parameters = np.zeros(
        (QUBITS, 2 * feature_count + 1),
        dtype=float,
    )

    parameters[:, :feature_count] = (
        0.04 * np.sin(indices + 1.0)
        / np.sqrt(feature_count)
    )
    parameters[:, feature_count:2 * feature_count] = (
        0.04 * np.cos(indices + 1.0)
        / np.sqrt(feature_count)
    )

    qubits = np.arange(QUBITS, dtype=float) + 1.0

    parameters[:, 0] = (
        np.pi / 2.0 + 0.07 * np.sin(qubits)
    )
    parameters[:, feature_count] = (
        0.50 + 0.08 * np.cos(qubits)
    )
    parameters[:, -1] = (
        0.35 + 0.06 * np.sin(qubits)
    )

    return parameters.ravel()


def unpack_parameters(theta, feature_count):
    theta = np.asarray(theta, dtype=float)

    check(
        theta.shape == (
            QUBITS * (2 * feature_count + 1),
        ),
        "Neispravna dimenzija parametara.",
    )

    matrix = theta.reshape(
        QUBITS,
        2 * feature_count + 1,
    )

    return (
        matrix[:, :feature_count],
        matrix[:, feature_count:2 * feature_count],
        matrix[:, -1],
    )


def qubit_model(theta, features):
    feature_count = features.shape[1]
    first_weights, phase_weights, final_angles = unpack_parameters(
        theta,
        feature_count,
    )

    first_angles = features @ first_weights.T
    phase_angles = features @ phase_weights.T

    first_cosine = np.cos(first_angles / 2.0)
    first_sine = np.sin(first_angles / 2.0)

    final_cosine = np.cos(final_angles / 2.0)[None, :]
    final_sine = np.sin(final_angles / 2.0)[None, :]

    phase_minus = np.exp(-0.5j * phase_angles)
    phase_plus = np.exp(0.5j * phase_angles)

    left = first_cosine * phase_minus
    right = first_sine * phase_plus

    amplitude_zero = (
        final_cosine * left - final_sine * right
    )
    amplitude_one = (
        final_sine * left + final_cosine * right
    )

    amplitudes = np.stack(
        (amplitude_zero, amplitude_one),
        axis=-1,
    )

    check(
        np.all(np.isfinite(amplitudes))
        and np.allclose(
            np.sum(np.abs(amplitudes)**2, axis=-1),
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "Jednokubitne amplitude nisu normalizovane.",
    )

    derivative_first = 0.5 * (
        -final_sine * first_sine * phase_minus
        + final_cosine * first_cosine * phase_plus
    )
    derivative_phase = 0.5j * (
        -final_sine * left + final_cosine * right
    )
    derivative_final = 0.5 * amplitude_zero

    # Izvodi verovatnoce |1> po uglovima a, b i c.
    derivatives = np.stack(
        (
            2.0 * np.real(
                amplitude_one.conj() * derivative_first
            ),
            2.0 * np.real(
                amplitude_one.conj() * derivative_phase
            ),
            2.0 * np.real(
                amplitude_one.conj() * derivative_final
            ),
        ),
        axis=-1,
    )

    return amplitudes, derivatives, (
        first_angles,
        phase_angles,
        final_angles,
    )


def product_distribution(amplitudes):
    batch = len(amplitudes)
    bit_probabilities = np.abs(amplitudes)**2

    factors = np.where(
        BITS.T[None, :, :] == 1,
        bit_probabilities[:, :, 1, None],
        bit_probabilities[:, :, 0, None],
    )

    prefix = np.ones(
        (batch, QUBITS + 1, DIM),
        dtype=float,
    )
    suffix = np.ones_like(prefix)

    prefix[:, 1:] = np.cumprod(
        factors,
        axis=1,
    )
    suffix[:, :-1] = np.cumprod(
        factors[:, ::-1, :],
        axis=1,
    )[:, ::-1, :]

    probabilities = prefix[:, -1]
    exclusion_products = (
        prefix[:, :-1] * suffix[:, 1:]
    )

    check(
        np.all(np.isfinite(probabilities))
        and np.allclose(
            probabilities.sum(axis=1),
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "Raspodela registra nije normalizovana.",
    )

    return probabilities, exclusion_products


def log_amplitudes(amplitudes):
    magnitudes = np.abs(amplitudes)
    result = np.full(magnitudes.shape, -np.inf)

    np.log(
        magnitudes,
        out=result,
        where=magnitudes > 0.0,
    )

    return 2.0 * result


def register_log_probabilities(theta, features):
    amplitudes, _, _ = qubit_model(theta, features)
    bit_logs = log_amplitudes(amplitudes)

    result = np.zeros(
        (len(features), DIM),
        dtype=float,
    )

    for qubit in range(QUBITS):
        result += np.where(
            BITS[:, qubit][None, :] == 1,
            bit_logs[:, qubit, 1, None],
            bit_logs[:, qubit, 0, None],
        )

    check(
        np.all(np.isfinite(result) | np.isneginf(result))
        and np.allclose(
            logsumexp(result, axis=1),
            0.0,
            atol=1e-10,
            rtol=0,
        ),
        "Logaritamska raspodela registra nije normalizovana.",
    )

    return result


def objective(theta, features, targets, weight):
    feature_count = features.shape[1]
    total = len(targets)

    check(
        total > 0
        and len(features) == total
        and np.all((targets >= 0) & (targets < N)),
        "Neispravan trening skup.",
    )

    gradient = np.zeros(
        (QUBITS, 2 * feature_count + 1),
        dtype=float,
    )
    loss = 0.0

    for start in range(0, total, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, total)
        batch_features = features[start:stop]
        batch_targets = targets[start:stop]

        amplitudes, derivatives, _ = qubit_model(
            theta,
            batch_features,
        )
        probabilities, exclusion = product_distribution(
            amplitudes
        )

        difference = probabilities.copy()
        difference[
            np.arange(stop - start),
            batch_targets,
        ] -= 1.0

        loss += float(np.sum(difference**2))

        bit_gradient = 2.0 * np.sum(
            difference[:, None, :]
            * exclusion
            * BIT_SIGNS[None, :, :],
            axis=2,
        )

        angle_gradient = (
            bit_gradient[:, :, None] * derivatives
        )

        gradient[:, :feature_count] += (
            angle_gradient[:, :, 0].T
            @ batch_features
        )
        gradient[:, feature_count:2 * feature_count] += (
            angle_gradient[:, :, 1].T
            @ batch_features
        )
        gradient[:, -1] += (
            angle_gradient[:, :, 2].sum(axis=0)
        )

    loss = loss / total + weight * np.dot(theta, theta)
    flat_gradient = (
        gradient.ravel() / total
        + 2.0 * weight * theta
    )

    check(
        np.isfinite(loss)
        and np.all(np.isfinite(flat_gradient)),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), flat_gradient


def fit_models(features, targets, weight, label):
    print(label)
    print("Trening prelaza:", len(targets))
    print("Regularizacija:", weight)

    feature_count = features.shape[1]
    models = []

    for slot in range(K):
        starting_point = initial_parameters(feature_count)
        best_loss = np.inf
        best_theta = starting_point.copy()

        def tracked_objective(candidate):
            nonlocal best_loss, best_theta

            loss, gradient = objective(
                candidate,
                features,
                targets[:, slot],
                weight,
            )

            if loss < best_loss:
                best_loss = loss
                best_theta = candidate.copy()

            return loss, gradient

        print(
            "Obuka registra:",
            slot + 1,
            flush=True,
        )

        result = minimize(
            tracked_objective,
            starting_point,
            method="CG",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
            },
        )

        chosen = best_theta.copy()
        loss, gradient = objective(
            chosen,
            features,
            targets[:, slot],
            weight,
        )
        gradient_norm = float(
            np.max(np.abs(gradient))
        )

        check(
            np.isfinite(best_loss)
            and np.all(np.isfinite(chosen)),
            "Optimizator nije sacuvao konacne parametre.",
        )

        print("Status optimizatora:", result.message)
        print("Sacuvani gubitak:", f"{loss:.8f}")
        print(
            "Norma gradijenta sacuvane tacke:",
            f"{gradient_norm:.8g}",
        )

        if (
            not result.success
            or gradient_norm > GRADIENT_TOLERANCE
        ):
            print(
                "UPOZORENJE: konvergencija nije potvrdjena."
            )

        models.append(chosen)

    return models


def predict_logs(models, features):
    check(
        len(models) == K,
        "Nedostaju modeli registara.",
    )

    result = np.empty(
        (len(features), K, DIM),
        dtype=float,
    )

    for slot, theta in enumerate(models):
        for start in range(0, len(features), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(features))

            result[start:stop, slot] = (
                register_log_probabilities(
                    theta,
                    features[start:stop],
                )
            )

    return result


def joint_distribution(log_weights):
    logs = np.asarray(log_weights, dtype=float).copy()

    check(
        logs.ndim == 2
        and np.all(np.isfinite(logs) | np.isneginf(logs)),
        "Neispravni logaritmi zajednicke raspodele.",
    )

    slots, states = logs.shape
    check(
        1 <= slots <= states,
        "Neispravne dimenzije zajednicke raspodele.",
    )

    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]

    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs[~allowed] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]

    maximum = logs[0].copy()
    parents = np.zeros(
        (slots, states),
        dtype=np.int64,
    )

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(
                increasing,
                forward[slot - 1, :, None],
                -np.inf,
            ),
            axis=0,
        )

        candidates = np.where(
            increasing,
            maximum[:, None],
            -np.inf,
        )
        parents[slot] = candidates.argmax(axis=0)
        maximum = logs[slot] + candidates[
            parents[slot],
            np.arange(states),
        ]

    log_partition = float(
        logsumexp(forward[-1])
    )

    if not np.isfinite(log_partition):
        raise DistributionError(
            "Kvantno stanje nema pozitivnu legalnu masu."
        )

    path = np.zeros(slots, dtype=np.int64)
    path[-1] = int(maximum.argmax())

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (
                    logs[slot + 1]
                    + backward[slot + 1]
                )[None, :],
                -np.inf,
            ),
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_partition
    )
    mode_log_mass = float(
        logs[np.arange(slots), path].sum()
    )

    check(
        np.all(np.diff(path) > 0)
        and np.isfinite(mode_log_mass)
        and mode_log_mass <= log_partition + 1e-8
        and log_partition <= 1e-8,
        "Neispravan maksimum zajednicke raspodele.",
    )
    check(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Zajednicka raspodela nije normalizovana.",
    )

    return {
        "path": path,
        "marginals": marginals,
        "log_partition": log_partition,
        "log_mode_probability": (
            mode_log_mass - log_partition
        ),
        "logs": logs,
    }


def evaluate(models, features, targets):
    check(
        len(features) == len(targets)
        and len(targets) > 0,
        "Neispravan evaluacioni skup.",
    )

    predicted_logs = predict_logs(models, features)
    losses = []
    hits = []

    for row, target in enumerate(targets):
        distribution = joint_distribution(
            predicted_logs[row, :, :N]
        )

        target_log_mass = float(
            distribution["logs"][
                np.arange(K), target
            ].sum()
        )

        loss = (
            distribution["log_partition"]
            - target_log_mass
        )

        check(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan evaluacioni NLL.",
        )

        losses.append(loss)
        hits.append(
            len(set(distribution["path"]) & set(target))
        )

    losses = np.asarray(losses, dtype=float)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(
            np.isposinf(losses).sum()
        ),
    }


def verify_qiskit(theta, features):
    amplitudes, _, angles = qubit_model(
        theta,
        features,
    )
    first_angles, phase_angles, final_angles = angles
    predicted_logs = register_log_probabilities(
        theta,
        features,
    )

    for row in range(len(features)):
        circuit = QuantumCircuit(QUBITS)

        for qubit in range(QUBITS):
            circuit.ry(
                float(first_angles[row, qubit]),
                qubit,
            )
            circuit.rz(
                float(phase_angles[row, qubit]),
                qubit,
            )
            circuit.ry(
                float(final_angles[qubit]),
                qubit,
            )

        check(
            all(
                item.operation.num_qubits == 1
                for item in circuit.data
            ),
            "Kolo sadrzi visekubitnu kapiju.",
        )

        actual = np.asarray(
            Statevector.from_instruction(circuit).data
        )

        expected = np.ones(DIM, dtype=complex)

        for qubit in range(QUBITS):
            expected *= amplitudes[
                row,
                qubit,
                BITS[:, qubit],
            ]

        check(
            np.allclose(
                actual,
                expected,
                atol=2e-11,
                rtol=2e-11,
            ),
            "Qiskit i algebraicke amplitude nisu usaglasene.",
        )
        check(
            np.allclose(
                np.abs(actual)**2,
                np.exp(predicted_logs[row]),
                atol=2e-11,
                rtol=0,
            ),
            "Qiskit i logaritamske verovatnoce nisu usaglasene.",
        )


def self_checks(data):
    sample_times = np.arange(
        1,
        min(len(data), 6),
    )
    sample_targets = data[sample_times, 0]

    for window in WINDOWS:
        features = history_features(
            data,
            sample_times,
            window,
        )
        theta = initial_parameters(features.shape[1])
        theta += 0.02 * np.sin(
            np.arange(len(theta)) + 1.0
        )

        verify_qiskit(theta, features[:2])

        _, analytic = objective(
            theta,
            features,
            sample_targets,
            0.001,
        )

        for index in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[index] = 1e-6

            numerical = (
                objective(
                    theta + offset,
                    features,
                    sample_targets,
                    0.001,
                )[0]
                - objective(
                    theta - offset,
                    features,
                    sample_targets,
                    0.001,
                )[0]
            ) / 2e-6

            check(
                np.isclose(
                    numerical,
                    analytic[index],
                    atol=2e-6,
                    rtol=2e-4,
                ),
                f"Gradijent nije potvrdjen za prozor "
                f"{window}, parametar {index}.",
            )

        print(
            "Provere kapija i gradijenta su prosle za prozor:",
            window,
            flush=True,
        )

    feature_count = 1 + 8 * WINDOWS[0]
    features = np.zeros((1, feature_count))
    features[:, 0] = 1.0

    phase_parameters = np.zeros(
        (QUBITS, 2 * feature_count + 1)
    )
    phase_parameters[0, 0] = np.pi / 2.0
    phase_parameters[0, -1] = np.pi / 2.0

    first_state = qubit_model(
        phase_parameters.ravel(),
        features,
    )[0]
    first_probability = float(
        np.abs(first_state[0, 0, 1])**2
    )

    phase_parameters[0, feature_count] = np.pi

    second_state = qubit_model(
        phase_parameters.ravel(),
        features,
    )[0]
    second_probability = float(
        np.abs(second_state[0, 0, 1])**2
    )

    check(
        first_probability > 1.0 - 1e-12
        and second_probability < 1e-12,
        "Provera fazne interferencije nije prosla.",
    )

    zero_parameters = np.zeros(
        QUBITS * (2 * feature_count + 1)
    )
    zero_logs = register_log_probabilities(
        zero_parameters,
        features,
    )
    check(
        np.isclose(zero_logs[0, 0], 0.0)
        and np.all(np.isneginf(zero_logs[0, 1:])),
        "Provera baznog stanja nije prosla.",
    )

    tiny_log = log_amplitudes(
        np.array([1e-180 + 0j])
    )[0]
    check(
        np.isfinite(tiny_log)
        and np.isclose(
            tiny_log,
            2.0 * np.log(1e-180),
            atol=1e-10,
            rtol=0,
        ),
        "Provera male amplitude nije prosla.",
    )

    toy = softmax(
        np.sin(np.arange(15).reshape(3, 5)),
        axis=1,
    )
    distribution = joint_distribution(np.log(toy))

    scores = [
        sum(
            np.log(toy[slot, combination[slot]])
            for slot in range(3)
        )
        for combination in itertools.combinations(range(5), 3)
    ]

    path = distribution["path"]
    mode_score = sum(
        np.log(toy[slot, path[slot]])
        for slot in range(3)
    )

    check(
        np.isclose(
            distribution["log_partition"],
            logsumexp(scores),
            atol=1e-10,
            rtol=0,
        )
        and np.isclose(
            mode_score,
            max(scores),
            atol=1e-10,
            rtol=0,
        ),
        "Iscrpna provera normalizacije i maksimuma nije prosla.",
    )

    single = np.full((3, 3), -np.inf)
    np.fill_diagonal(single, 0.0)
    single_result = joint_distribution(single)

    check(
        np.array_equal(
            single_result["path"],
            np.arange(3),
        )
        and np.isclose(
            single_result["log_partition"],
            0.0,
            atol=1e-12,
            rtol=0,
        ),
        "Provera jedinog legalnog stanja nije prosla.",
    )

    try:
        joint_distribution(
            np.full((3, 3), -np.inf)
        )
    except DistributionError:
        pass
    else:
        raise RuntimeError(
            "Nulta legalna masa nije pravilno odbijena."
        )

    print("Sve ugradjene pocetne provere su prosle.")


def print_metrics(label, metrics):
    print(label)
    print("Broj kola:", len(metrics["hits"]))
    print("Srednji NLL:", f'{metrics["nll"]:.8f}')
    print(
        "Srednji broj pogodaka:",
        f'{metrics["hits"].mean():.6f}',
    )
    print(
        "Nulte ciljne verovatnoce:",
        metrics["zero_targets"],
    )
    print(
        "Broj kola po pogocima od 0 do 7:",
        np.bincount(
            metrics["hits"],
            minlength=K + 1,
        ).tolist(),
    )


def process_dataset(dataset):
    path = dataset["path"]
    data = dataset["data"]

    targets = data[1:]
    times = np.arange(1, len(data))
    total = len(targets)

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(
        VALIDATION_END_FRACTION * total
    )

    check(
        0 < train_end < validation_end < total,
        "Neispravna hronoloska podela.",
    )

    print()
    print("CSV:", path.name)
    print("Redova:", len(data))
    print("SHA256:", dataset["digest"])
    print("Trening prelaza:", train_end)
    print(
        "Validacionih prelaza:",
        validation_end - train_end,
    )
    print("Test prelaza:", total - validation_end)

    best_settings = None
    best_validation = np.inf

    for window in WINDOWS:
        features = history_features(
            data,
            times,
            window,
        )

        for weight in REGULARIZATIONS:
            print()
            print(
                "Kandidat:",
                "prozor",
                window,
                "regularizacija",
                weight,
            )

            models = fit_models(
                features[:train_end],
                targets[:train_end],
                weight,
                "UCENJE ZA VALIDACIJU",
            )

            try:
                metrics = evaluate(
                    models,
                    features[train_end:validation_end],
                    targets[train_end:validation_end],
                )
            except DistributionError as error:
                print(
                    "Kandidat nema definisanu legalnu raspodelu."
                )
                print(str(error))
                continue

            print_metrics(
                "VALIDACIONI REZULTAT",
                metrics,
            )

            if (
                np.isfinite(metrics["nll"])
                and metrics["nll"] < best_validation
            ):
                best_validation = metrics["nll"]
                best_settings = (window, weight)

    check(
        best_settings is not None,
        "Nijedan kandidat nema konacan validacioni NLL.",
    )

    window, weight = best_settings
    print()
    print("Izabrani prozor:", window)
    print("Izabrana regularizacija:", weight)

    features = history_features(
        data,
        times,
        window,
    )

    test_models = fit_models(
        features[:validation_end],
        targets[:validation_end],
        weight,
        "UCENJE ZA ODVOJENI TEST",
    )

    test_metrics = evaluate(
        test_models,
        features[validation_end:],
        targets[validation_end:],
    )

    print_metrics(
        "ODVOJENI TEST",
        test_metrics,
    )

    # Parametri su zamrznuti tokom testa.
    # Ranija test kola mogu biti istorijski ulaz kasnijeg kola.
    # Test rezultat ne bira hiperparametre.

    final_models = fit_models(
        features,
        targets,
        weight,
        "ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA",
    )

    next_features = history_features(
        data,
        np.array([len(data)]),
        window,
    )

    for theta in final_models:
        verify_qiskit(theta, next_features)

    print("Zavrsne Qiskit provere svih sedam registara su prosle.")

    next_logs = predict_logs(
        final_models,
        next_features,
    )[0]

    distribution = joint_distribution(
        next_logs[:, :N]
    )

    prediction = distribution["path"]
    combination = prediction + 1

    print(
        "Log legalne mase:",
        f'{distribution["log_partition"]:.8f}',
    )
    print(
        "Log modelske verovatnoce NEXT:",
        f'{distribution["log_mode_probability"]:.8f}',
    )
    print(
        "Najvece marginalne verovatnoce:",
        " ".join(
            f"{value:.5f}"
            for value in distribution["marginals"].max(axis=1)
        ),
    )

    probe_times = np.unique(
        np.array([
            min(window, len(data)),
            len(data) // 2,
            len(data),
        ], dtype=int)
    )
    probe_features = history_features(
        data,
        probe_times,
        window,
    )
    probe_probabilities = np.exp(
        predict_logs(final_models, probe_features)
    )

    variation = float(np.max(np.abs(
        probe_probabilities
        - np.exp(next_logs)[None, :, :]
    )))

    print(
        "Promena raspodele izmedju stvarnih istorijskih konteksta:",
        f"{variation:.10g}",
    )

    if np.array_equal(prediction, data[-1]):
        print("PROVERA: NEXT ponavlja poslednje kolo.")

    longest_run = 1
    current_run = 1

    for difference in np.diff(combination):
        if difference == 1:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 1

    if longest_run >= 5:
        print(
            "PROVERA: najmanje pet uzastopnih brojeva."
        )

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    span_percentile = float(
        100.0 * np.mean(historical_spans <= span)
    )

    print("Raspon NEXT kombinacije:", span)
    print(
        "Udeo istorijskih raspona manjih ili jednakih NEXT rasponu:",
        f"{span_percentile:.3f}%",
    )

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": np.exp(next_logs),
        "marginals": distribution["marginals"],
    }


def main():
    print(TITLE)
    print("Qiskit:", qiskit.__version__)
    print("Sedam registara sa po sest kubita.")
    print("Svaki kubit koristi RY, RZ i RY.")
    print("Ucenje raspodele preko kvantnih amplituda.")
    print("Normalizacija i maksimum svih legalnih kombinacija.")
    print("NLL je negativni logaritam modelske verovatnoce stvarnog kola.")

    datasets = [
        load_csv(path)
        for path in FILES
    ]

    self_checks(datasets[0]["data"])

    results = [
        process_dataset(dataset)
        for dataset in datasets
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print("PROVERA: dva ulaza imaju isti SHA256.")
            print(first["name"])
            print(second["name"])

        if np.array_equal(
            first["next"],
            second["next"],
        ):
            print()
            print("PROVERA ISTIH NEXT KOMBINACIJA")
            print(first["name"])
            print(second["name"])
            print(
                "Najveca razlika sirovih raspodela:",
                float(np.max(np.abs(
                    first["probabilities"]
                    - second["probabilities"]
                ))),
            )
            print(
                "Najveca razlika legalnih marginalnih raspodela:",
                float(np.max(np.abs(
                    first["marginals"]
                    - second["marginals"]
                ))),
            )

    print()
    print("ZAVRSNE NEXT KOMBINACIJE")

    for result in results:
        print(result["name"])
        print(
            "NEXT:",
            " ".join(map(str, result["next"])),
        )


if __name__ == "__main__":
    main()



"""
Quantum Single Qubit Gates
Qiskit: 1.4.4
Sedam registara sa po sest kubita.
Svaki kubit koristi RY, RZ i RY.
Ucenje raspodele preko kvantnih amplituda.
Normalizacija i maksimum svih legalnih kombinacija.
NLL je negativni logaritam modelske verovatnoce stvarnog kola.
Provere kapija i gradijenta su prosle za prozor: 1
Provere kapija i gradijenta su prosle za prozor: 3
Sve ugradjene pocetne provere su prosle.

CSV: loto7_4716_k89.csv
Redova: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih prelaza: 707
Test prelaza: 708

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89972650
Norma gradijenta sacuvane tacke: 9.0430735e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95320047
Norma gradijenta sacuvane tacke: 9.9355619e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96825375
Norma gradijenta sacuvane tacke: 8.9348001e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96636655
Norma gradijenta sacuvane tacke: 6.4696385e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96022253
Norma gradijenta sacuvane tacke: 7.5470914e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97448037
Norma gradijenta sacuvane tacke: 5.4207611e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92207762
Norma gradijenta sacuvane tacke: 5.4561282e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 20.86432307
Srednji broj pogodaka: 1.264498
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [157, 292, 183, 65, 9, 1, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91391003
Norma gradijenta sacuvane tacke: 8.541995e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97454007
Norma gradijenta sacuvane tacke: 6.6007181e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99179939
Norma gradijenta sacuvane tacke: 8.3039222e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99128854
Norma gradijenta sacuvane tacke: 6.5821445e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98964351
Norma gradijenta sacuvane tacke: 7.3869528e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00688435
Norma gradijenta sacuvane tacke: 9.783452e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95004652
Norma gradijenta sacuvane tacke: 4.7344208e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 22.79351137
Srednji broj pogodaka: 1.263083
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [145, 309, 182, 66, 4, 0, 1, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89919370
Norma gradijenta sacuvane tacke: 8.2985882e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95241979
Norma gradijenta sacuvane tacke: 7.0485836e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96750601
Norma gradijenta sacuvane tacke: 6.4974728e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96578881
Norma gradijenta sacuvane tacke: 9.8453712e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95944725
Norma gradijenta sacuvane tacke: 9.873855e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97363247
Norma gradijenta sacuvane tacke: 7.2436255e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92111974
Norma gradijenta sacuvane tacke: 8.0251901e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 20.91041602
Srednji broj pogodaka: 1.224894
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [160, 305, 173, 62, 6, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 3300
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91234655
Norma gradijenta sacuvane tacke: 5.676638e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97231304
Norma gradijenta sacuvane tacke: 8.9814844e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99009095
Norma gradijenta sacuvane tacke: 5.6066481e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98964687
Norma gradijenta sacuvane tacke: 7.7144742e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98808263
Norma gradijenta sacuvane tacke: 2.5162691e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00543111
Norma gradijenta sacuvane tacke: 2.9971759e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94754284
Norma gradijenta sacuvane tacke: 2.0767278e-06
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 22.27394481
Srednji broj pogodaka: 1.250354
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [158, 297, 183, 56, 12, 1, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 4007
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90047570
Norma gradijenta sacuvane tacke: 9.7924481e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95341052
Norma gradijenta sacuvane tacke: 8.3981581e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96824720
Norma gradijenta sacuvane tacke: 8.4875004e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96680757
Norma gradijenta sacuvane tacke: 6.429161e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96082589
Norma gradijenta sacuvane tacke: 9.5254431e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97466715
Norma gradijenta sacuvane tacke: 9.8554558e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92113497
Norma gradijenta sacuvane tacke: 8.2350784e-06
ODVOJENI TEST
Broj kola: 708
Srednji NLL: 20.85072745
Srednji broj pogodaka: 1.216102
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [176, 290, 168, 63, 9, 2, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 4715
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90069493
Norma gradijenta sacuvane tacke: 9.9461591e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95325069
Norma gradijenta sacuvane tacke: 5.7635536e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96850365
Norma gradijenta sacuvane tacke: 6.4642013e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96720693
Norma gradijenta sacuvane tacke: 4.1537481e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96061034
Norma gradijenta sacuvane tacke: 5.7684168e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97520395
Norma gradijenta sacuvane tacke: 9.0505681e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92141582
Norma gradijenta sacuvane tacke: 7.4714638e-06
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -4.06627789
Log modelske verovatnoce NEXT: -13.13506133
Najvece marginalne verovatnoce: 0.29520 0.11831 0.13578 0.14926 0.10955 0.23904 0.21130
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.01478144637
Raspon NEXT kombinacije: 36
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 92.430%

CSV: loto7_4716_k89_loto_2980.csv
Redova: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih prelaza: 447
Test prelaza: 447

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90018480
Norma gradijenta sacuvane tacke: 9.2355529e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95253382
Norma gradijenta sacuvane tacke: 6.1890121e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96833701
Norma gradijenta sacuvane tacke: 9.5997489e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96613270
Norma gradijenta sacuvane tacke: 9.8735966e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95989637
Norma gradijenta sacuvane tacke: 9.3050947e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97468053
Norma gradijenta sacuvane tacke: 3.5400171e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92212008
Norma gradijenta sacuvane tacke: 8.5719713e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 20.87451945
Srednji broj pogodaka: 1.275168
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [87, 197, 120, 39, 4, 0, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91466202
Norma gradijenta sacuvane tacke: 9.3974194e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97408169
Norma gradijenta sacuvane tacke: 9.8525499e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99199939
Norma gradijenta sacuvane tacke: 5.59194e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99118025
Norma gradijenta sacuvane tacke: 7.8981666e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98942395
Norma gradijenta sacuvane tacke: 4.8942403e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00672699
Norma gradijenta sacuvane tacke: 7.3401764e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95077018
Norma gradijenta sacuvane tacke: 7.5911447e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 22.14325053
Srednji broj pogodaka: 1.275168
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [94, 183, 125, 43, 2, 0, 0, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89920073
Norma gradijenta sacuvane tacke: 6.8646628e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95141068
Norma gradijenta sacuvane tacke: 8.1244619e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96780778
Norma gradijenta sacuvane tacke: 6.9058533e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96570020
Norma gradijenta sacuvane tacke: 5.9873796e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95889426
Norma gradijenta sacuvane tacke: 4.4870851e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97370416
Norma gradijenta sacuvane tacke: 7.6226213e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92070414
Norma gradijenta sacuvane tacke: 8.8518262e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 20.93195463
Srednji broj pogodaka: 1.230425
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [91, 212, 101, 37, 5, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 2084
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91272097
Norma gradijenta sacuvane tacke: 6.3324892e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97180801
Norma gradijenta sacuvane tacke: 8.7344429e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99050950
Norma gradijenta sacuvane tacke: 4.4243481e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98994006
Norma gradijenta sacuvane tacke: 6.249927e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98796283
Norma gradijenta sacuvane tacke: 3.4824105e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00558750
Norma gradijenta sacuvane tacke: 4.5265533e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94805242
Norma gradijenta sacuvane tacke: 9.7739511e-06
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 21.89787986
Srednji broj pogodaka: 1.257271
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [100, 180, 123, 40, 4, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 2531
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90186144
Norma gradijenta sacuvane tacke: 5.8248777e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95318735
Norma gradijenta sacuvane tacke: 7.0224711e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96798397
Norma gradijenta sacuvane tacke: 9.6274887e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96645855
Norma gradijenta sacuvane tacke: 6.0133992e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96004910
Norma gradijenta sacuvane tacke: 7.7425149e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97458403
Norma gradijenta sacuvane tacke: 8.0462399e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92201987
Norma gradijenta sacuvane tacke: 9.2894343e-06
ODVOJENI TEST
Broj kola: 447
Srednji NLL: 20.82051981
Srednji broj pogodaka: 1.210291
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [103, 184, 128, 28, 3, 1, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 2978
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90251353
Norma gradijenta sacuvane tacke: 8.8990267e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95310036
Norma gradijenta sacuvane tacke: 8.3054666e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96845029
Norma gradijenta sacuvane tacke: 9.5497327e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96715825
Norma gradijenta sacuvane tacke: 8.244208e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96003764
Norma gradijenta sacuvane tacke: 8.1420353e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97507339
Norma gradijenta sacuvane tacke: 7.6885329e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92126182
Norma gradijenta sacuvane tacke: 8.8107129e-06
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -4.28715475
Log modelske verovatnoce NEXT: -12.99829530
Najvece marginalne verovatnoce: 0.28784 0.12246 0.11690 0.15795 0.11701 0.19737 0.22168
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.02745657753
Raspon NEXT kombinacije: 36
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 92.414%

CSV: loto7_4716_k89_loto_plus_1736.csv
Redova: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih prelaza: 260
Test prelaza: 261

Kandidat: prozor 1 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89494538
Norma gradijenta sacuvane tacke: 8.3643765e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95310402
Norma gradijenta sacuvane tacke: 5.3810211e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96748355
Norma gradijenta sacuvane tacke: 8.2126264e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96657859
Norma gradijenta sacuvane tacke: 8.5005125e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96033334
Norma gradijenta sacuvane tacke: 8.1336488e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97320231
Norma gradijenta sacuvane tacke: 9.6344765e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91992040
Norma gradijenta sacuvane tacke: 5.9136275e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 21.14858501
Srednji broj pogodaka: 1.188462
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [63, 109, 67, 18, 3, 0, 0, 0]

Kandidat: prozor 1 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91022132
Norma gradijenta sacuvane tacke: 6.6758559e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97407809
Norma gradijenta sacuvane tacke: 9.8562501e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99128689
Norma gradijenta sacuvane tacke: 9.5696036e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99109082
Norma gradijenta sacuvane tacke: 6.3793228e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99036650
Norma gradijenta sacuvane tacke: 7.1461474e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00639666
Norma gradijenta sacuvane tacke: 7.1922915e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94827606
Norma gradijenta sacuvane tacke: 6.7551007e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 21.94353684
Srednji broj pogodaka: 1.234615
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [61, 107, 64, 26, 2, 0, 0, 0]

Kandidat: prozor 3 regularizacija 0.001
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89337107
Norma gradijenta sacuvane tacke: 8.8606415e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95179089
Norma gradijenta sacuvane tacke: 9.843031e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96615142
Norma gradijenta sacuvane tacke: 9.078008e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96577545
Norma gradijenta sacuvane tacke: 9.7115791e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95909363
Norma gradijenta sacuvane tacke: 9.2045989e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97217229
Norma gradijenta sacuvane tacke: 9.2276012e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.91777437
Norma gradijenta sacuvane tacke: 8.3999873e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 21.32376588
Srednji broj pogodaka: 1.273077
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [57, 106, 71, 22, 3, 1, 0, 0]

Kandidat: prozor 3 regularizacija 0.01
UCENJE ZA VALIDACIJU
Trening prelaza: 1213
Regularizacija: 0.01
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.90847727
Norma gradijenta sacuvane tacke: 7.6618448e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97226130
Norma gradijenta sacuvane tacke: 7.7466699e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98936152
Norma gradijenta sacuvane tacke: 4.0114609e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.99005946
Norma gradijenta sacuvane tacke: 4.495583e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.98856023
Norma gradijenta sacuvane tacke: 2.2746199e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 1.00477863
Norma gradijenta sacuvane tacke: 7.9741326e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.94536067
Norma gradijenta sacuvane tacke: 6.962313e-06
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 22.24952072
Srednji broj pogodaka: 1.165385
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [60, 118, 64, 15, 3, 0, 0, 0]

Izabrani prozor: 1
Izabrana regularizacija: 0.001
UCENJE ZA ODVOJENI TEST
Trening prelaza: 1473
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89683011
Norma gradijenta sacuvane tacke: 6.5826423e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95293662
Norma gradijenta sacuvane tacke: 8.1404497e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96784506
Norma gradijenta sacuvane tacke: 9.2862609e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96678169
Norma gradijenta sacuvane tacke: 9.5650245e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96030036
Norma gradijenta sacuvane tacke: 4.469218e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97408516
Norma gradijenta sacuvane tacke: 8.6966291e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92216670
Norma gradijenta sacuvane tacke: 6.7486589e-06
ODVOJENI TEST
Broj kola: 261
Srednji NLL: 20.66323705
Srednji broj pogodaka: 1.199234
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [63, 107, 68, 22, 1, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Trening prelaza: 1734
Regularizacija: 0.001
Obuka registra: 1
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.89634171
Norma gradijenta sacuvane tacke: 5.1236167e-06
Obuka registra: 2
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95257153
Norma gradijenta sacuvane tacke: 8.9073931e-06
Obuka registra: 3
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96802078
Norma gradijenta sacuvane tacke: 9.2767984e-06
Obuka registra: 4
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.96688416
Norma gradijenta sacuvane tacke: 9.6990307e-06
Obuka registra: 5
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.95996586
Norma gradijenta sacuvane tacke: 6.6368515e-06
Obuka registra: 6
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.97428867
Norma gradijenta sacuvane tacke: 4.7162166e-06
Obuka registra: 7
Status optimizatora: Optimization terminated successfully.
Sacuvani gubitak: 0.92191625
Norma gradijenta sacuvane tacke: 9.7962904e-06
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -4.06443058
Log modelske verovatnoce NEXT: -13.05628044
Najvece marginalne verovatnoce: 0.30971 0.11316 0.13277 0.13666 0.11895 0.25524 0.19954
Promena raspodele izmedju stvarnih istorijskih konteksta: 0.02445009821
Raspon NEXT kombinacije: 36
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 92.450%


ZAVRSNE NEXT KOMBINACIJE

loto7_4716_k89.csv
NEXT: 1 x 16 y 29 z 37

loto7_4716_k89_loto_2980.csv
NEXT: 1 x 13 y 29 z 37

loto7_4716_k89_loto_plus_1736.csv
NEXT: 1 x 15 y 25 z 37
"""





"""
Q6-Q10	                            Osnovna ideja	                                                        Primena
Single Qubit Gates	                Rotacije, relativne faze i promene merne baze.	                        Učenje parametara jednokubitnih kapija. 
Multiple Qubits and Entanglement	Tenzorski proizvod, kontrolisane kapije i zapletenost.	                Učenje zapletenih stanja unutar šestokubitnih registara. 
Shor	                            Modularna eksponencijacija, procena faze i nalaženje reda.	            Posebna matematička veza sa istorijom kola. 
Bernstein-Vazirani	                Oporavak skrivenog niza iz linearnog binarnog orakla.	                Orakl izveden iz podataka koji zadovoljava taj uslov. 
Simon	                            Otkrivanje skrivene XOR simetrije funkcije.	                            Funkcija sa odgovarajućom strukturom preslikavanja. 


 6. Quantum_Single_Qubit_Gates.py
 7. Quantum_Entanglement.py
 8. Shor
 9. Bernstein-Vazirani
10. Simon

Za poslednja tri prvo treba definisati orakl iz istorije i proveriti uslove izvornog algoritma. 



--------
6. 

Quantum_Single_Qubit_Gates

Jednokubitna rotacija i fazna interferencija. 
Svaki šestokubitni registar ima proizvoljno stanje. 
Parametri kapija iz istorijskih prelaza kroz regresiju raspodele.



-------------
7. 

Quantum_Entanglement

Kontrolisane kapije i zapletenost. 
Između slojeva naučenih rotacija koriste se kontrolisane operacije. 
Provera redukovanih stanja pokazuje da li kolo zaista stvara zapletenost.

Rotacije zavisne od istorije pripremaju stanje, zatim CNOT lanac povezuje šest kubita, a završne rotacije menjaju mernu bazu. 
Provera zapletenosti preko čistoće redukovanih stanja i poređenje amplituda sa Qiskitom pre obuke.

Koristi naučene rotacije i CNOT lanac. 
Zapletenost se proverava unutar svakog šestokubitnog registra preko redukovanih matrica gustine. 
Redosled bitova prati Qiskit 1.4.

- smer i inverziju CNOT permutacija
- kompleksni gradijent kroz RY, RZ i CNOT
- faktor gradijenta Brierovog gubitka
- Bell i GHZ provere
- čistoća redukovanih stanja i Qiskit redosled bitova
- logaritamska normalizacija, globalni maksimum i hronološka podela 


Svaki kubit koristi niz RY → RZ → RY, koji pretvara relativnu fazu u promenu izlaznih verovatnoća. 
Prozor istorije bira se validacijom. 
Konačna kombinacija dobija se kao globalni maksimum raspodele nad svim legalnim izborima sedam brojeva.

Koristim sedam šestokubitnih registara i niz RY → RZ → RY na svakom kubitu. 
Prozore od jednog i tri prethodna kola biram validacijom. 
Optimizator uči parametre kapija iz cele trening istorije.


- izvodi verovatnoća po sva tri ugla RY → RZ → RY
- prefiksni i sufiksni proizvodi u gradijentu
- dimenzije parametara za oba prozora istorije
- redosled bitova prema Qiskitu
- logaritamska normalizacija i globalni izbor maksimuma
- granice treninga, validacije, testa i završnog učenja.



-------------
8. 

Shorov deo modularna eksponencijacija, QPE, inverzna QFT, procena faze i verizni ralomci za nalaženje reda. 

Orakl je \(f_a(x)=a^x\bmod 41\), uz izbor baze a hronološkom validacijom. 
Modul 41 omogućava modularni inverz za svaki broj od 1 do 39. 
QPE odredjuje red, a dodatni model uči amplitude koraka po modularnoj orbiti. 
Superpozicija periodičnih sopstvenih stanja modularnog operatora, čiju dimenziju određuje red dobijen QPE postupkom. 
Provera Shorove faktorizacije broja 15. 
Nalaženje reda koristiće 12 pomoćnih kubita i šest radnih kubita, a poređenje sa Qiskitom manju QPE konstrukciju.

Prediktivna dopuna uči superpoziciju periodičnih sopstvenih stanja modularnog operatora. 
Koristi modul 41, poslednje kolo kao početak orbite i celu istoriju prelaza za obuku. 
Bazu i broj koeficijenata bira validacijom.

QPE simulira 12 pomoćnih i šest radnih kubita. 
Manja QPE konstrukcija poredi se sa Qiskitom. 
Prediktivni deo koristi sedam šestokubitnih registara.



-------------
9.

Quantum_Bernstein_Vazirani 

Orakl, fazni povratni uticaj i Hadamard transformacije. 
Hadamard transformacije, linearni BV orakl i fazni povratni uticaj. 
Za Loto učim dodatnu fazu orakla iz istorijskih prelaza. 
Prethodno kolo je poznati ulaz. Buduća kombinacija se ne ugrađuje u orakl.

Koristim sedam registara sa po šest kubita i po jedan pomoćni kubit. 
Registri se simuliraju zasebno. 



----------------
10. 

Quantum_Simon

Sačuvaću orakl sa XOR periodom i rešavanje jednačina nad GF(2). 

Amplitude na parovima stanja \(x\) i \(x \oplus b\). 
Kandidati za XOR period biraju se iz treninga, a konačni model hronološkom validacijom. 
Simonovo kolo proverava period konstruisanog orakla. 

Dvostruki registar, orakl sa XOR periodom, Hadamard transformacije i rešavanje jednačina nad GF(2). 
Amplitude na Simonovim parovima stanja. 
Kandidati za period procenjuju se na treningu, a konačni model bira hronološkom validacijom.

Simonova provera koristi 12 kubita. 
Predikcija koristi sedam zasebnih registara sa po šest kubita. 
"""
