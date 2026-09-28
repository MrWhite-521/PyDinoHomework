import pygame
import os
import random
import numpy as np
import pickle
import time
from PIL import Image

pygame.init()
SCREEN_HEIGHT, SCREEN_WIDTH = 600, 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Dino Evolution AI")

# ================= ЗАГРУЗКА СПРАЙТОВ =================
def load_img(path):
    try:
        return pygame.image.load(os.path.join("Assets", path)).convert_alpha()
    except Exception as e:
        print(f"Ошибка загрузки {path}: {e}")
        surf = pygame.Surface((40, 40))
        surf.fill((255, 0, 0))
        return surf

RUNNING = [load_img("Dino/DinoRun1.png"), load_img("Dino/DinoRun2.png")]
JUMPING = load_img("Dino/DinoJump.png")
DUCKING = [load_img("Dino/DinoDuck1.png"), load_img("Dino/DinoDuck2.png")]
SMALL_CACTUS = [load_img("Cactus/SmallCactus1.png"), load_img("Cactus/SmallCactus2.png"), load_img("Cactus/SmallCactus3.png")]
LARGE_CACTUS = [load_img("Cactus/LargeCactus1.png"), load_img("Cactus/LargeCactus2.png"), load_img("Cactus/LargeCactus3.png")]
BIRD = [load_img("Bird/Bird1.png"), load_img("Bird/Bird2.png")]
CLOUD = load_img("Other/Cloud.png")
BG = load_img("Other/Track.png")

# ================= ЗАГРУЗКА GIF И WOLL =================
def load_gif_frames(path):
    """Загружает все кадры GIF в формате RGBA (сохраняет прозрачность)."""
    frames = []
    try:
        pil_image = Image.open(path)
        for frame in range(pil_image.n_frames):
            pil_image.seek(frame)
            # Конвертация в RGBA — гарантирует правильные цвета и прозрачность
            rgba = pil_image.convert("RGBA")
            frame_surface = pygame.image.fromstring(
                rgba.tobytes(), rgba.size, "RGBA"
            ).convert_alpha()
            frames.append(frame_surface)
    except Exception as e:
        print(f"Ошибка GIF {path}: {e}")
    return frames

# Загружаем анимацию огня
FIRE_FRAMES = load_gif_frames(
    os.path.join("Assets", "additional_obstacles", "fire_and_fireball.gif")
)
if not FIRE_FRAMES:
    print("Внимание: GIF не загружен, используется заглушка.")
    _stub = pygame.Surface((40, 40), pygame.SRCALPHA)
    _stub.fill((255, 100, 0, 255))
    FIRE_FRAMES = [_stub]

# Загружаем стену
try:
    WOLL_IMG = pygame.image.load(
        os.path.join("Assets", "additional_obstacles", "woll.png")
    ).convert_alpha()
except Exception as e:
    print(f"Ошибка загрузки woll.png: {e}")
    WOLL_IMG = pygame.Surface((40, 40), pygame.SRCALPHA)
    WOLL_IMG.fill((100, 100, 100, 255))

# ================= НЕЙРОСЕТЬ =================
class NeuralNetwork:
    def __init__(self, input_size=6, hidden_size=10, output_size=4):
        self.w1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.w2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5

    def predict(self, state):
        z1 = np.dot(state, self.w1) + self.b1
        hidden = np.tanh(z1)
        z2 = np.dot(hidden, self.w2) + self.b2
        return np.argmax(z2)

    def mutate(self, rate=0.2, strength=0.5):
        child = NeuralNetwork()
        for attr in ['w1', 'b1', 'w2', 'b2']:
            param = getattr(self, attr)
            mask = np.random.random(param.shape) < rate
            noise = np.random.randn(*param.shape) * strength
            setattr(child, attr, param + mask * noise)
        return child

    def copy(self):
        child = NeuralNetwork()
        child.w1, child.b1, child.w2, child.b2 = (
            self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()
        )
        return child

# ================= ДИНОЗАВРИК =================
class Dinosaur:
    X_POS, Y_POS, Y_POS_DUCK, JUMP_VEL, GROUND_Y = 80, 310, 340, 8.5, 310

    def __init__(self, brain=None):
        self.duck_img, self.run_img, self.jump_img = DUCKING, RUNNING, JUMPING
        self.dino_duck = self.dino_jump = False
        self.dino_run = True
        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x, self.dino_rect.y = self.X_POS, self.Y_POS
        self.brain = brain if brain else NeuralNetwork()
        self.fitness = 0
        self.alive = True
        # Заряд
        self.has_charge = False
        self.charge_icon = (
            pygame.transform.scale(FIRE_FRAMES[0], (30, 30))
            if FIRE_FRAMES else None
        )

    def update(self, action):
        if action in [1, 2] and self.dino_jump:
            action = 0

        fireball_to_return = None
        if action == 3 and self.has_charge:
            # ВЫСТРЕЛ: повёрнут на 90°, летит быстро вправо
            fireball_to_return = Fireball(
                self.dino_rect.x + self.dino_rect.width,
                self.dino_rect.y + 20,
                rotated=True,
                speed=25
            )
            self.has_charge = False
            action = 0

        if action == 1 and not self.dino_jump:
            self.dino_duck, self.dino_run, self.dino_jump = False, False, True
        elif action == 2 and not self.dino_jump:
            self.dino_duck, self.dino_run, self.dino_jump = True, False, False
        elif not self.dino_jump:
            self.dino_duck, self.dino_run, self.dino_jump = False, True, False

        if self.dino_duck:
            self.image = self.duck_img[self.step_index // 5]
            self.dino_rect.y = self.Y_POS_DUCK
            self.step_index += 1
        elif self.dino_run:
            self.image = self.run_img[self.step_index // 5]
            self.dino_rect.y = self.Y_POS
            self.step_index += 1
        elif self.dino_jump:
            self.image = self.jump_img
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
            if self.dino_rect.y >= self.GROUND_Y:
                self.dino_rect.y, self.dino_jump, self.jump_vel = (
                    self.GROUND_Y, False, self.JUMP_VEL
                )

        if self.step_index >= 10:
            self.step_index = 0
        self.fitness += 1
        return fireball_to_return

    def draw(self):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))
        if self.has_charge and self.charge_icon:
            SCREEN.blit(self.charge_icon, (10, 10))

    def get_state(self, nearest, game_speed):
        if nearest:
            distance = nearest.rect.x - self.dino_rect.x
            height = nearest.rect.y
            if isinstance(nearest, Bird):
                obs_type = 1
            elif isinstance(nearest, WollWall):
                obs_type = 2
            elif isinstance(nearest, Fireball):
                obs_type = 3
            else:
                obs_type = 0
        else:
            distance, height, obs_type = SCREEN_WIDTH, 0, 0

        return np.array([
            max(0, min(1, distance / 500)),
            max(0, min(1, height / 400)),
            max(0, min(1, game_speed / 30)),
            obs_type / 3.0,
            1 if self.dino_jump else 0,
            1 if self.has_charge else 0
        ])

    def get_action(self, state):
        return self.brain.predict(state)

# ================= ПРЕПЯТСТВИЯ =================
class SmallCactus:
    def __init__(self):
        self.image = SMALL_CACTUS[random.randint(0, 2)]
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = SCREEN_WIDTH, 325
    def update(self, speed): self.rect.x -= speed
    def draw(self): SCREEN.blit(self.image, self.rect)

class LargeCactus:
    def __init__(self):
        self.image = LARGE_CACTUS[random.randint(0, 2)]
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = SCREEN_WIDTH, 300
    def update(self, speed): self.rect.x -= speed
    def draw(self): SCREEN.blit(self.image, self.rect)

class Bird:
    def __init__(self):
        self.image = BIRD
        self.rect = self.image[0].get_rect()
        self.rect.x, self.rect.y = SCREEN_WIDTH, 250
        self.index = 0
    def update(self, speed): self.rect.x -= speed
    def draw(self):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index // 5], self.rect)
        self.index += 1

class WollWall:
    def __init__(self):
        self.image = pygame.transform.scale(WOLL_IMG, (80, 200))
        self.rect = self.image.get_rect()
        self.rect.x = SCREEN_WIDTH
        self.rect.y = 380 - self.rect.height
    def update(self, speed): self.rect.x -= speed
    def draw(self): SCREEN.blit(self.image, self.rect)

class Fireball:
    def __init__(self, x, y, rotated=False, speed=-20):
        base_frame = pygame.transform.scale(FIRE_FRAMES[0], (80, 80))
        if rotated:
            base_frame = pygame.transform.rotate(base_frame, 90)
        self.image = base_frame
        self.rect = self.image.get_rect()
        self.rect.x, self.rect.y = x, y
        self.speed = speed
        self.alive = True

    def update(self):
        self.rect.x += self.speed
        if self.rect.x > SCREEN_WIDTH or self.rect.x < -self.rect.width:
            self.alive = False

    def draw(self):
        SCREEN.blit(self.image, (self.rect.x, self.rect.y))

class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()
    def update(self, speed):
        self.x -= speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)
    def draw(self): SCREEN.blit(self.image, (self.x, self.y))

class Background:
    def __init__(self):
        self.x, self.y, self.speed = 0, 380, 20
    def update(self):
        self.x -= self.speed
        if self.x <= -BG.get_width():
            self.x = 0
    def draw(self):
        SCREEN.blit(BG, (self.x, self.y))
        SCREEN.blit(BG, (self.x + BG.get_width(), self.y))

# ================= ИГРОВОЙ ЦИКЛ =================
def run_game(brain, visual=True, generation=0, individual=0):
    game_speed = 20
    player = Dinosaur(brain)
    obstacles, fireballs = [], []
    cloud, bg = Cloud(), Background()
    bg.speed = game_speed
    frames_without_obstacle = 0
    clock = pygame.time.Clock()
    fire_spawn_timer, fire_spawn_delay = 0, random.randint(150, 300)

    while player.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return 0
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return player.fitness

        # Периодический спавн подбираемого заряда
        fire_spawn_timer += 1
        has_fireball_on_screen = any(isinstance(o, Fireball) for o in obstacles)
        if (fire_spawn_timer > fire_spawn_delay
                and not player.has_charge
                and not has_fireball_on_screen):
            fp = Fireball(SCREEN_WIDTH, 310, rotated=False, speed=-game_speed)
            obstacles.append(fp)
            fire_spawn_timer, fire_spawn_delay = 0, random.randint(200, 400)

        # Спавн препятствий
        if len(obstacles) == 0 and frames_without_obstacle > random.randint(60, 100):
            rand = random.randint(0, 3)
            if rand == 0:
                obstacles.append(SmallCactus())
            elif rand == 1:
                obstacles.append(LargeCactus())
            elif rand == 2:
                obstacles.append(Bird())
            else:
                # Перед стеной ОБЯЗАТЕЛЬНО спавним заряд ближе к игроку
                if not player.has_charge:
                    fp = Fireball(
                        SCREEN_WIDTH - 550, 310,
                        rotated=False, speed=-game_speed
                    )
                    obstacles.append(fp)
                obstacles.append(WollWall())
            frames_without_obstacle = 0

        nearest = obstacles[0] if obstacles else None
        state = player.get_state(nearest, game_speed)
        action = player.get_action(state)

        new_fb = player.update(action)
        if new_fb:
            fireballs.append(new_fb)

        bg.update()
        cloud.update(game_speed)

        # Обновление снарядов (выстрелов)
        for fb in fireballs[:]:
            fb.update()
            if not fb.alive:
                fireballs.remove(fb)
                continue
            for obs in obstacles[:]:
                if isinstance(obs, WollWall) and fb.rect.colliderect(obs.rect):
                    obstacles.remove(obs)
                    fireballs.remove(fb)
                    break

        # Обновление препятствий
        for obs in obstacles[:]:
            if isinstance(obs, Fireball):
                obs.update()
                if player.dino_rect.colliderect(obs.rect):
                    player.has_charge = True
                    obstacles.remove(obs)
                    continue
            else:
                obs.update(game_speed)
                if player.dino_rect.colliderect(obs.rect):
                    player.alive = False
                    break

            if obs.rect.x < -obs.rect.width:
                if obs in obstacles:
                    obstacles.remove(obs)

        frames_without_obstacle += 1
        if player.fitness >= 10000:
            player.alive = False
            break

        if visual:
            SCREEN.fill((255, 255, 255))
            bg.draw()
            cloud.draw()
            player.draw()
            for obs in obstacles:
                obs.draw()
            for fb in fireballs:
                fb.draw()

            font = pygame.font.Font('freesansbold.ttf', 20)
            SCREEN.blit(font.render(f"Score: {player.fitness}", True, (0, 0, 0)),
                        (SCREEN_WIDTH - 150, 30))
            SCREEN.blit(font.render(f"Gen: {generation}", True, (0, 0, 0)),
                        (SCREEN_WIDTH - 150, 55))
            SCREEN.blit(font.render(f"Ind: {individual}", True, (0, 0, 0)),
                        (SCREEN_WIDTH - 150, 80))

            pygame.display.update()
            clock.tick(60)
        else:
            clock.tick(60)

    return player.fitness

# ================= ЭВОЛЮЦИЯ =================
class EvolutionTrainer:
    def __init__(self, population_size=10, continue_training=False):
        self.population_size = population_size
        self.best_score = 0
        self.perfect_score = 10000

        if continue_training:
            try:
                self.best_brain = load_brain()
                self.population = [self.best_brain.copy()] + [
                    self.best_brain.mutate(0.3, 0.7)
                    for _ in range(population_size - 1)
                ]
                try:
                    with open('training_stats.pkl', 'rb') as f:
                        self.best_score = pickle.load(f).get('best_score', 0)
                except:
                    pass
            except FileNotFoundError:
                self.population = [NeuralNetwork() for _ in range(population_size)]
                self.best_brain = None
        else:
            self.population = [NeuralNetwork() for _ in range(population_size)]
            self.best_brain = None

    def evaluate(self, gen):
        results = []
        for i, brain in enumerate(self.population):
            print(f"Gen {gen}, Ind {i+1}/{self.population_size}")
            score = run_game(brain, True, gen, i+1)
            results.append((brain, score))
            print(f"  Score: {score}")
            if score >= self.perfect_score:
                save_brain(brain, f"perfect_dino_{int(time.time())}.pkl")
                save_brain(brain, "best_dino.pkl")
                save_training_stats(score, gen, i+1)
                return results, True
        results.sort(key=lambda x: x[1], reverse=True)
        self.population = [b for b, _ in results]
        if results[0][1] > self.best_score:
            self.best_score = results[0][1]
            self.best_brain = results[0][0].copy()
            save_brain(self.best_brain)
            save_training_stats(self.best_score, gen)
        return results, False

    def evolve(self):
        new_pop = [self.population[0].copy()]
        while len(new_pop) < self.population_size:
            new_pop.append(random.choice(self.population[:3]).mutate(0.2, 0.5))
        self.population = new_pop

    def train(self, generations=10):
        for gen in range(generations):
            print(f"\n--- ПОКОЛЕНИЕ {gen+1}/{generations} ---")
            results, completed = self.evaluate(gen + 1)
            if completed:
                return self.best_brain
            self.evolve()
        return self.best_brain

# ================= СОХРАНЕНИЕ =================
def save_training_stats(score, gen, ind=None):
    with open('training_stats.pkl', 'wb') as f:
        pickle.dump({
            'best_score': score, 'generation': gen, 'individual': ind,
            'date': time.strftime('%Y-%m-%d %H:%M:%S')
        }, f)

def load_training_stats():
    try:
        with open('training_stats.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

def save_brain(brain, filename="best_dino.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(brain, f)

def load_brain(filename="best_dino.pkl"):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# ================= МЕНЮ =================
def main_menu():
    clock = pygame.time.Clock()
    font = pygame.font.Font('freesansbold.ttf', 30)
    small_font = pygame.font.Font('freesansbold.ttf', 20)
    running = True

    while running:
        SCREEN.fill((255, 255, 255))
        SCREEN.blit(font.render("DINO EVOLUTION AI", True, (0, 0, 0)),
                    (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 200))

        try:
            load_brain()
            stats = load_training_stats()
            status = f"✓ Модель обучена (рекорд: {stats['best_score']})" if stats else "✓ Модель найдена"
            color = (0, 150, 0)
        except:
            status, color = "✗ Модель не обучена", (150, 0, 0)

        SCREEN.blit(small_font.render(status, True, color),
                    (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 160))

        items = [
            ("1 - НОВОЕ ОБУЧЕНИЕ", SCREEN_HEIGHT // 2 - 100),
            ("2 - ДООБУЧИТЬ МОДЕЛЬ", SCREEN_HEIGHT // 2 - 50),
            ("3 - ЗАПУСТИТЬ ЛУЧШЕГО ИИ", SCREEN_HEIGHT // 2),
            ("4 - ВЫЙТИ", SCREEN_HEIGHT // 2 + 50)
        ]
        for text, y in items:
            SCREEN.blit(font.render(text, True, (0, 0, 0)),
                        (SCREEN_WIDTH // 2 - 150, y))

        SCREEN.blit(RUNNING[0], (SCREEN_WIDTH // 2 - 20, SCREEN_HEIGHT // 2 + 120))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    trainer = EvolutionTrainer(10, False)
                    save_brain(trainer.train(10))
                elif event.key == pygame.K_2:
                    try:
                        pop = int(input("Размер популяции (Enter=10): ") or "10")
                        gens = int(input("Поколений (Enter=10): ") or "10")
                    except:
                        pop, gens = 10, 10
                    trainer = EvolutionTrainer(pop, True)
                    save_brain(trainer.train(gens))
                elif event.key == pygame.K_3:
                    try:
                        brain = load_brain()
                        run_game(brain, True, "BEST", "AI")
                    except:
                        print("Сначала обучите модель!")
                elif event.key == pygame.K_4:
                    running = False
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main_menu()