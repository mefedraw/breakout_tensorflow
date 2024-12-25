import pygame
import sys
from settings import *
import numpy as np
from ai_agent import DQNAgent


class BreakoutGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Breakout")
        self.clock = pygame.time.Clock()
        self.running = True

        self.reset_game()

    def reset_game(self):
        """
        Сброс состояния игры для начала нового эпизода.
        """
        self.running = True
        self.paddle_x = WIDTH // 2 - PADDLE_WIDTH // 2
        self.paddle_y = HEIGHT - 30
        self.ball_x = WIDTH // 2
        self.ball_y = HEIGHT // 2
        self.ball_dx = 4
        self.ball_dy = -4
        self.blocks = self.create_blocks()

        # Инициализируем предыдущее расстояние между платформой и мячом
        self.prev_distance = abs((self.paddle_x + PADDLE_WIDTH / 2) - self.ball_x)

    def calculate_reward(self, prev_blocks_len):
        """
        Рассчитывает награду за текущий шаг игры.
        """
        reward = 0
        done = False

        # Если мяч падает ниже экрана (проигрыш)
        if self.ball_y > HEIGHT:
            reward = -2  # Штраф за проигрыш
            done = True

        # Если количество блоков уменьшилось
        elif prev_blocks_len > len(self.blocks):
            reward += 5  # Награда за разрушение блока

        # Дополнительная награда за приближение платформы к мячу
        current_distance = abs((self.paddle_x + PADDLE_WIDTH / 2) - self.ball_x)
        reward += (self.prev_distance - current_distance) * 0.01  # Небольшой коэффициент

        # Обновляем предыдущее расстояние
        self.prev_distance = current_distance

        return reward, done

    def create_blocks(self):
        """
        Создаёт блоки для игры.
        """
        blocks = []
        for row in range(BLOCK_ROWS):
            for col in range(BLOCK_COLS):
                blocks.append(pygame.Rect(col * BLOCK_WIDTH, row * BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT))
        return blocks

    def handle_input(self):
        """
        Обрабатывает пользовательский ввод (клавиши влево и вправо).
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.paddle_x > 0:
            self.paddle_x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT] and self.paddle_x < WIDTH - PADDLE_WIDTH:
            self.paddle_x += PADDLE_SPEED

    def update_ball(self):
        """
        Обновляет позицию мяча и обрабатывает отскоки от стен.
        """
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Отскок от вертикальных стен
        if self.ball_x <= 0 or self.ball_x >= WIDTH - BALL_RADIUS:
            self.ball_dx *= -1

        # Отскок от верхней стенки
        if self.ball_y <= 0:
            self.ball_dy *= -1

    def check_collision(self):
        """
        Проверяет столкновения мяча с платформой и блоками.
        """
        # Отскок от платформы
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, BALL_RADIUS * 2, BALL_RADIUS * 2)
        if paddle_rect.colliderect(ball_rect):
            self.ball_dy = -abs(self.ball_dy)
            hit_pos = (self.ball_x + BALL_RADIUS) - (self.paddle_x + PADDLE_WIDTH // 2)
            self.ball_dx += hit_pos // (PADDLE_WIDTH // 10)
            self.ball_dx = max(min(self.ball_dx, 7), -7)

        # Удар по блокам
        for block in self.blocks[:]:
            if block.colliderect(ball_rect):
                self.blocks.remove(block)
                self.ball_dy *= -1
                break

    def render(self):
        """
        Отрисовывает все элементы игры на экране.
        """
        self.screen.fill(BLACK)

        # Рисуем платформу
        paddle_rect = pygame.Rect(self.paddle_x, self.paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT)
        pygame.draw.rect(self.screen, WHITE, paddle_rect)

        # Рисуем мяч
        pygame.draw.circle(self.screen, BLUE, (self.ball_x, self.ball_y), BALL_RADIUS)

        # Рисуем блоки
        for block in self.blocks:
            pygame.draw.rect(self.screen, RED, block)

        pygame.display.flip()

    def get_state(self):
        """
        Возвращает текущее состояние игры для агента.
        """
        # Нормализуем данные для состояния
        return np.array([
            self.paddle_x / WIDTH,
            self.ball_x / WIDTH,
            self.ball_y / HEIGHT,
            self.ball_dx / 5,
            self.ball_dy / 5,
            (self.ball_x - (self.paddle_x + PADDLE_WIDTH / 2)) / WIDTH  # Разница по оси X
        ])

    def train(self):
        """
        Метод для обучения агента.
        """
        # Параметры агента
        state_size = 6  # paddle_x, ball_x, ball_y, ball_dx, ball_dy, delta_x
        action_size = 3  # Действия: влево, вправо, стоять
        agent = DQNAgent(state_size, action_size)
        episodes = 1000
        batch_size = 32
        target_update_freq = 10  # Частота обновления целевой сети

        for e in range(episodes):
            self.reset_game()  # Сброс состояния игры
            state = self.get_state()
            state = np.reshape(state, [1, state_size])
            total_reward = -5.0

            # Начальное количество блоков
            prev_blocks_len = len(self.blocks)

            while self.running:
                # Агент выбирает действие
                action = agent.act(state)

                # Применяем действие
                if action == 0 and self.paddle_x > 0:
                    self.paddle_x -= PADDLE_SPEED
                elif action == 1 and self.paddle_x < WIDTH - PADDLE_WIDTH:
                    self.paddle_x += PADDLE_SPEED

                # Обновляем состояние игры
                self.update_ball()
                self.check_collision()

                # Рассчитываем награду
                next_state = self.get_state()
                next_state = np.reshape(next_state, [1, state_size])
                reward, done = self.calculate_reward(prev_blocks_len)

                # Обновляем количество блоков
                prev_blocks_len = len(self.blocks)

                # Сохраняем опыт агента
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if done:
                    print(f"Эпизод: {e + 1}/{episodes}, Награда: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
                    break

                # Обновляем экран для визуализации (можно отключить для ускорения обучения)
                # self.render()
                # self.clock.tick(FPS)

            # Обучение агента
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Обновление целевой сети
            if e % target_update_freq == 0:
                agent.update_target_network()

            # Сохранение модели каждые 100 эпизодов
            if (e + 1) % 100 == 0:
                agent.model.save(f"dqn_model_episode_{e + 1}.h5")
                print(f"Модель сохранена после эпизода {e + 1}")
