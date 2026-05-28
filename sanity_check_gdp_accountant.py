from utils.dpsgd_utils import compute_dpdr_gdp_mu, gdp_epsilon_from_mu


def main():
    cases = [
        {
            "name": "MNIST",
            "dataset_size": 60000,
            "batch_size": 256,
            "epochs": 20,
            "sigma_g": 0.803,
            "sigma_perp": 0.81,
            "sigma_alpha": 2.0,
        },
        {
            "name": "CIFAR10",
            "dataset_size": 50000,
            "batch_size": 256,
            "epochs": 20,
            "sigma_g": 0.835,
            "sigma_perp": 0.84,
            "sigma_alpha": 3.0,
        },
        {
            "name": "SVHN",
            "dataset_size": 73000,
            "batch_size": 256,
            "epochs": 20,
            "sigma_g": 0.695,
            "sigma_perp": 0.696,
            "sigma_alpha": 2.0,
        },
    ]

    delta = 1e-5
    steps_dr = 50
    steps_interval = 40000

    for case in cases:
        details = compute_dpdr_gdp_mu(
            local_dataset_size=case["dataset_size"],
            local_batch_size=case["batch_size"],
            epochs=case["epochs"],
            sigma_perp=case["sigma_perp"],
            sigma_alpha=case["sigma_alpha"],
            sigma_g=case["sigma_g"],
            steps_dr=steps_dr,
            steps_interval=steps_interval,
            return_details=True,
        )
        epsilon = gdp_epsilon_from_mu(details["mu"], delta)
        print(
            "{name}: corrected GDP epsilon={epsilon:.6f}, mu={mu:.6f}, "
            "T={total_steps}, T_gdr={gdr_steps}, T_sgd={sgd_steps}".format(
                name=case["name"],
                epsilon=epsilon,
                **details,
            )
        )


if __name__ == "__main__":
    main()
