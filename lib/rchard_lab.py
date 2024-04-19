import time
import requests
import math
import numpy as np
import matplotlib.pyplot as plt


class DriveHandler:
    def __init__(self, robotino_ip, params={'sid': 'example_circle'}):
        self.OMNIDRIVE_PATH = "/data/omnidrive"
        self.ODOMETRY = "/data/odometry"
        self.omnidrive_url = "http://" + robotino_ip + self.OMNIDRIVE_PATH
        self.params = params
        self.odometry_url = "http://" + robotino_ip + self.ODOMETRY

    def set_speed(self, x):
        vel_vect = x
        request_result = requests.post(url=self.omnidrive_url, params=self.params, json=vel_vect)
        if request_result.status_code != requests.codes.ok:
            raise RuntimeError("Error: post to %s with params %s failed", self.omnidrive_url, self.params)

    def odometry(self):
        request_result = requests.get(url=self.odometry_url, params=self.params)
        if request_result.status_code != requests.codes.ok:
            raise RuntimeError("Error: post to %s with params %s failed", self.omnidrive_url, self.params)
        else:
            data = request_result.json()
            print(f'Odometer data {data}')
            return data

    def set_movement_1(self, movement):
        start_time = time.time()
        time1 = start_time
        odoms = []
        while time.time() - start_time < movement[0]:
            if time.time() - time1 > 0.2:
                odom1 = self.odometry()
                odoms.append(odom1)
                time1 = time.time()
            self.set_speed(movement[1])
        return odoms

    def circle_trajectory(self, amp, R):
        vels = []
        vx = amp
        wz = vx / R
        time = 2 * math.pi / wz
        vels.append([time, [vx, 0, wz]])
        return vels

    def lab4(self, trajectory, trajectory_pid):
        odom_start = self.odometry()
        traj = []
        for movement in trajectory:
            traj.extend(self.set_movement_1(movement))

        for movement in trajectory:
            traj.extend(self.set_movement_1(movement))

        # Параметры окружности
        radius = 0.3
        theta = np.linspace(0, 2 * np.pi, 100)

        # Координаты для окружности
        x_circle = radius * np.cos(theta)
        y_circle = radius * np.sin(theta)

        # Отображение точек на графике
        points = np.array([[odom[0] - odom_start[0], odom[1] - odom_start[1]] for odom in traj])

        # Вычисление центра траектории
        center = np.mean(points, axis=0)

        # Вычисление угла поворота
        angle = np.arctan2(points[-1][1] - center[1], points[-1][0] - center[0])

        # Применение поворота и сдвига к точкам траектории
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        shifted_points = points - center
        rotated_points = np.dot(shifted_points, rotation_matrix)

        # Создание графика
        plt.figure(figsize=(6, 6))
        plt.plot(x_circle, y_circle, 'b', label='Desired Trajectory')  # Окружность
        plt.plot(rotated_points[:, 0], rotated_points[:, 1], 'r', label='Trajectory without PID')  # Траектория робота
        plt.plot(rotated_points[:, 0], rotated_points[:, 1], 'g', label='Trajectory with PID')  # Траектория робота

        # plt.plot(x_circle, y_circle, 'b', label='Desired Trajectory')  # Окружность
        # plt.plot(points_without_pid[:, 0], points_without_pid[:, 1], 'r',
        #          label='Trajectory without PID')  # Траектория робота без PID
        # plt.plot(points_with_pid[:, 0], points_with_pid[:, 1], 'g',
        #          label='Trajectory with PID')  # Траектория робота с PID

        # Настройка графика
        plt.title('График окружности с траекторией робота')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')  # Установка одинакового масштаба осей

        plt.show()


driver = DriveHandler("192.168.0.1")
speed_amplitude = 0.1  # Амплитуда скорости
trajectory = driver.circle_trajectory(speed_amplitude, 0.3)
driver.lab4(trajectory)
