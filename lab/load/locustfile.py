"""Locust load: browse 60% / order 25% / pay 15%, wave-shaped (low, normal, peak)."""
from locust import HttpUser, LoadTestShape, between, task


class Shopper(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(60)
    def browse(self):
        self.client.get("/products")

    @task(25)
    def order(self):
        self.client.post("/orders")

    @task(15)
    def pay(self):
        self.client.post("/pay")


class Waves(LoadTestShape):
    stages = [(300, 5), (600, 20), (900, 50), (1200, 20)]  # (until_s, users)

    def tick(self):
        t = self.get_run_time() % 1200
        for until, users in self.stages:
            if t < until:
                return users, users
        return 20, 20
