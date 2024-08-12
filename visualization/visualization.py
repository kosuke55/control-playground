import multiprocessing
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.spatial import ConvexHull


class Visualization:
    def __init__(
        self,
        controller_mode,
        states,
        estimated_states,
        observed_states,
        control_inputs,
        delayed_inputs,
        diff_history,
        success_time,
        time,
        L1,
        L2,
        f1,
        f2,
        initial_state,
        dt,
        control_dt,
        dead_time,
        add_measurement_noise,
        use_estimates,
        use_quantize,
        encoder_resolution,
        friction_compensations,
        save_dir=None,
        save_format="gif",
    ):
        self.controller_mode = controller_mode
        self.states = states
        self.estimated_states = estimated_states
        self.observed_states = observed_states
        self.control_inputs = control_inputs
        self.delayed_inputs = delayed_inputs
        self.diff_history = diff_history
        self.success_time = success_time
        self.time = time
        self.L1 = L1
        self.L2 = L2
        self.f1 = f1
        self.f2 = f2
        self.initial_state = initial_state
        self.dt = dt
        self.control_dt = control_dt
        self.dead_time = dead_time
        self.add_measurement_noise = add_measurement_noise
        self.use_estimates = use_estimates
        self.use_quantize = use_quantize
        self.encoder_resolution = encoder_resolution
        self.friction_compensations = friction_compensations
        self.save_dir = save_dir
        self.save_format = save_format

    def save_animation(self, ani):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        Writer = animation.writers["ffmpeg"]
        writer = Writer(fps=15, metadata=dict(artist="Me"), bitrate=1800)
        initial_state = np.round(np.rad2deg(self.initial_state), 1)
        save_file = f"{self.save_dir}/{self.controller_mode}_θ-{initial_state[0]}_{initial_state[2]}_dt-{self.dt}-{self.control_dt}_dead-{self.dead_time}_f-{self.f1}-{self.f2}_n-{self.add_measurement_noise}_e-{self.use_estimates}_q-{self.use_quantize}-{self.encoder_resolution}.mp4"
        print(f"Animation saving... {save_file}")
        ani.save(f"{save_file}", writer=writer)

        print("Animation saved.")

    def animate(self):
        fig = plt.figure(figsize=(10, 13))
        fig.suptitle(f"Controller mode: {self.controller_mode}", fontsize=16)
        fig.subplots_adjust(top=0.95)
        gs = GridSpec(5, 1, figure=fig, height_ratios=[5, 2, 2, 2, 2])
        gs.update(hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3])
        ax5 = fig.add_subplot(gs[4])

        # Adjust the width of ax2, ax3, ax4, ax5
        for ax in [ax2, ax3, ax4, ax5]:
            pos1 = ax.get_position()
            width = pos1.width
            center = pos1.x0 + (pos1.width - width) / 2
            pos2 = [center, pos1.y0, width, pos1.height]
            ax.set_position(pos2)

        ax1.set_xlim(-4.5, 4.5)
        ax1.set_ylim(-4.5, 4.5)
        ax1.set_aspect("equal")
        ax1.grid()
        (line,) = ax1.plot([], [], "o-", lw=2, c="cornflowerblue", label="Actual")
        (line_est,) = ax1.plot([], [], "x--", lw=2, c="orange", label="Estimated")
        time_template = "time = %.1fs"
        time_text = ax1.text(0.08, 0.9, "", transform=ax1.transAxes)

        info_text = (
            f"f1: {self.f1}\n"
            f"f2: {self.f2}\n"
            f"Initial State: {np.round(np.rad2deg(self.initial_state),1)}\n"
            f"simulation dt: {self.dt}\n"
            f"control dt: {self.control_dt}\n"
            f"Dead Time: {self.dead_time}\n"
            f"Measurement Noise: {self.add_measurement_noise}\n"
            f"Use Estimates: {self.use_estimates}\n"
            f"Use Quantize: {self.use_quantize}\n"
            f"Encoder Resolution: {self.encoder_resolution}"
        )
        ax1.text(
            0.08,
            0.28,
            info_text,
            transform=ax1.transAxes,
            fontsize=10,
            verticalalignment="center",
        )

        ax1.legend()
        ax2.plot(self.time, self.control_inputs, c="cornflowerblue", label="Control input")
        ax2.plot(
            self.time,
            self.delayed_inputs,
            label="Delayed input",
            c="sandybrown",
            linestyle="--",
        )
        if len(self.friction_compensations) > 0:
            ax2.plot(
                self.time,
                self.friction_compensations,
                label="Friction compensation",
                c="brown",
                linestyle="--",
                linewidth=1,
            )
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Input")
        ax2.legend()
        ax2.grid()
        control_time_bar = ax2.axvline(x=0, color="r", linestyle="--")
        control_text = ax2.text(0.5, 0.2, "", transform=ax2.transAxes)

        ax3.plot(
            self.time,
            self.observed_states[:, 0],
            c="lightgreen",
            label="Observed θ1",
            linestyle="--",
        )
        ax3.plot(self.time, self.states[:, 0], c="cornflowerblue", label="True θ1")
        ax3.plot(
            self.time,
            self.estimated_states[:, 0],
            c="orange",
            label="Estimated θ1",
            linestyle="--",
        )
        ax3.legend()
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Angle [rad]")
        theta1_time_bar = ax3.axvline(x=0, color="r", linestyle="--")
        theta1_text = ax3.text(0.5, 0.2, "", transform=ax3.transAxes)

        if self.observed_states.shape[1] == 4:
            ax4.plot(
                self.time,
                self.observed_states[:, 2],
                c="lightgreen",
                label="Observed θ2",
                linestyle="--",
            )
        else:
            ax4.plot(
                self.time,
                self.observed_states[:, 1],
                c="lightgreen",
                label="Observed θ2",
                linestyle="--",
            )
        ax4.plot(self.time, self.states[:, 2], c="cornflowerblue", label="True θ2")
        ax4.plot(
            self.time,
            self.estimated_states[:, 2],
            c="orange",
            label="Estimated θ2",
            linestyle="--",
        )
        ax4.legend()
        ax4.set_xlabel("Time [s]")
        ax4.set_ylabel("Angle [rad]")
        theta2_time_bar = ax4.axvline(x=0, color="r", linestyle="--")
        theta2_text = ax4.text(0.5, 0.2, "", transform=ax4.transAxes)

        # visualize diff_history
        ax5.plot(self.time, self.diff_history, c="cornflowerblue", label="diff")
        if self.success_time is not None:
            ax5.axvline(x=self.success_time, color="r", linestyle="--", label="Success")
            # show success time in the plot
            ax5.text(
                self.success_time,
                0.5,
                str(self.success_time),
            )

        ax5.set_xlabel("Time [s]")
        ax5.set_ylabel("diff")
        ax5.legend()
        ax5.grid()

        def init():
            line.set_data([], [])
            line_est.set_data([], [])
            time_text.set_text("")
            control_time_bar.set_xdata([0])
            theta1_time_bar.set_xdata([0])
            theta2_time_bar.set_xdata([0])
            return (
                line,
                line_est,
                time_text,
                control_time_bar,
                theta1_time_bar,
                theta2_time_bar,
            )

        def update(i):
            x1 = self.L1 * np.sin(self.states[i, 0])
            y1 = self.L1 * np.cos(self.states[i, 0])
            x2 = x1 + self.L2 * np.sin(self.states[i, 2])
            y2 = y1 + self.L2 * np.cos(self.states[i, 2])

            x1_est = self.L1 * np.sin(self.estimated_states[i, 0])
            y1_est = self.L1 * np.cos(self.estimated_states[i, 0])
            x2_est = x1_est + self.L2 * np.sin(self.estimated_states[i, 2])
            y2_est = y1_est + self.L2 * np.cos(self.estimated_states[i, 2])

            line.set_data([0, x1, x2], [0, y1, y2])
            line_est.set_data([0, x1_est, x2_est], [0, y1_est, y2_est])
            time_text.set_text(time_template % self.time[i])
            control_text.set_text(
                f"u = {np.round(self.control_inputs[i], 2)},\nu_delayed = {np.round(self.delayed_inputs[i],2)}"
            )
            control_time_bar.set_xdata([self.time[i]])
            theta1_time_bar.set_xdata([self.time[i]])
            theta1_text.set_text(
                f"θ1 = {np.round(self.states[i, 0], 2)},\nθ1_est = {np.round(self.estimated_states[i, 0], 2)}"
            )
            theta2_time_bar.set_xdata([self.time[i]])
            theta2_text.set_text(
                f"θ2 = {np.round(self.states[i, 2], 2)},\nθ2_est = {np.round(self.estimated_states[i, 2], 2)}"
            )
            return (
                line,
                line_est,
                time_text,
                control_time_bar,
                control_text,
                theta1_time_bar,
                theta1_text,
                theta2_time_bar,
                theta2_text,
            )

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.time),
            interval=(self.time[1] - self.time[0]) * 1000,
            init_func=init,
            blit=True,
        )

        # save animation
        if self.save_dir:
            save_process = multiprocessing.Process(target=self.save_animation, args=(ani,))
            save_process.start()

        plt.show()


def visualize_roa(
    controller_mode,
    total_trials,
    theta1_range,
    theta2_range,
    successful_states,
    unsuccessful_states,
    processing_time_list,
    success_time_list,
    show_plot,
    save_dir,
):
    successful_states = np.array(successful_states)
    unsuccessful_states = np.array(unsuccessful_states)
    average_processing_time = np.mean(processing_time_list)
    average_success_time = np.mean(success_time_list)
    plt.figure(figsize=(10, 8))
    if len(successful_states) > 0:
        plt.scatter(
            np.degrees(successful_states[:, 0]),
            np.degrees(successful_states[:, 1]),
            color="green",
            label="Successful",
        )
        if len(successful_states) > 2 and len(np.unique(successful_states[:, 0])) > 1:
            hull = ConvexHull(successful_states)
            for simplex in hull.simplices:
                plt.plot(
                    np.degrees(successful_states[simplex, 0]),
                    np.degrees(successful_states[simplex, 1]),
                    "k-",
                )
    if len(unsuccessful_states) > 0:
        plt.scatter(
            np.degrees(unsuccessful_states[:, 0]),
            np.degrees(unsuccessful_states[:, 1]),
            color="red",
            label="Unsuccessful",
        )
    plt.xlabel("Theta1 (degrees)")
    plt.ylabel("Theta2 (degrees)")
    plt.legend()
    plt.title(f"ROA for {controller_mode} controller")
    plt.text(
        0.02,
        0.95,
        f"Average proccesing time: {average_processing_time:.2f} seconds\nAverage success time: {average_success_time}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    ROA = str(len(successful_states)) + "/" + str(total_trials)
    plt.text(
        0.02,
        0.85,
        f"ROA: {ROA}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7),
    )
    plt.grid(True)
    plt.xlim([0, np.degrees(theta1_range[-1])])
    plt.ylim([0, np.degrees(theta2_range[-1])])
    if show_plot:
        plt.show()
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(f"{save_dir}/ROA_{controller_mode}.png")
    print(f"ROA for {controller_mode} controller: {ROA}")


def visualize_poles(poles):
    plt.figure()
    plt.scatter(np.real(poles), np.imag(poles), marker="x", label="Poles")
    plt.axhline(0, color="black", lw=0.5)
    plt.axvline(0, color="black", lw=0.5)
    plt.title("Pole-Zero Map")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()
    plt.show()
