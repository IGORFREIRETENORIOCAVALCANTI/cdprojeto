from matplotlib import pyplot as plt


def plot_confidence_interval(confidence_intervals, title, confidence_str, values_str, x_ticks):
    uniform_x_values = range(len(x_ticks))

    i = 0
    for name in confidence_intervals:
        upper_bound = confidence_intervals[name][1]
        bottom_bound = confidence_intervals[name][0]
        mean_point_bound = (upper_bound + bottom_bound) / 2

        bounds_size = [[mean_point_bound - bottom_bound], [upper_bound - mean_point_bound]]
        plt.errorbar(
            uniform_x_values[i],
            mean_point_bound,
            yerr=bounds_size,
            fmt='.',
            capsize=5,
            label=f'interval: {name} {bottom_bound:.2f}, {upper_bound:.2f}'
        )

        i += 1

    plt.xticks(uniform_x_values, labels=[str(val) for val in x_ticks])
    plt.xlabel(f'{title}')
    plt.ylabel(f'{values_str}')
    plt.title(f'{title} with {confidence_str} confidence')
    plt.legend()
    plt.grid(True)
    plt.figure(figsize=(12, 10))

    plt.show()
