import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration
        
        # Discretización del espacio de estados
        self.buckets = (
            np.linspace(-1.2, 0.6, 20),    # Posición del carro
            np.linspace(-0.07, 0.07, 20)   # Velocidad del carro
        )
        
        # Inicializar Q-table con valores cero
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
        # Métricas
        self.rewards = []
        self.episode_lengths = []
        
    def discretize_state(self, state):
        """Convierte estado continuo a discreto"""
        return tuple(np.digitize(s, bucket) for s, bucket in zip(state, self.buckets))
    
    def epsilon_greedy_action(self, state):
        """Política epsilon-greedy para selección de acciones"""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()  # Exploración
        return np.argmax(self.q_table[state])      # Explotación
    
    def qlearning_update(self, state, action, reward, next_state, done):
        """Actualización Q-Learning: Q(s,a) = Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]"""
        current_q = self.q_table[state][action]
        
        if done:
            # Si el episodio terminó, no hay estado siguiente
            td_target = reward
        else:
            # Q-LEARNING: Usa el MÁXIMO Q-value del siguiente estado (política greedy)
            max_next_q = np.max(self.q_table[next_state])
            td_target = reward + self.gamma * max_next_q
        
        # Error TD y actualización
        td_error = td_target - current_q
        self.q_table[state][action] += self.lr * td_error
    
    def train(self, episodes=1000, max_steps=200):
        """Entrenamiento usando algoritmo Q-Learning"""
        for episode in range(episodes):
            # Reiniciar entorno
            state, _ = self.env.reset()
            state = self.discretize_state(state)
            
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Elegir acción usando política ε-greedy (para exploración)
                action = self.epsilon_greedy_action(state)
                
                # Ejecutar acción, observar reward y siguiente estado
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self.discretize_state(next_state)
                
                # Recompensa adicional por progreso hacia la meta
                if next_state[0] > state[0]:  # Movimiento hacia la derecha
                    reward += 0.1
                
                done = terminated or truncated
                
                # *** CLAVE Q-LEARNING: Actualizar usando el MÁXIMO del siguiente estado ***
                # No necesitamos elegir la siguiente acción aquí
                self.qlearning_update(state, action, reward, next_state, done)
                
                # Transición al siguiente estado
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Guardar métricas
            self.rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Decay de epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Progreso cada 100 episodios
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.rewards[-100:])
                avg_steps = np.mean(self.episode_lengths[-100:])
                print(f"Episodio {episode+1}: Reward promedio: {avg_reward:.2f}, "
                      f"Pasos promedio: {avg_steps:.1f}, Epsilon: {self.epsilon:.3f}")
    
    def test_agent(self, episodes=5, render=True):
        """Probar el agente entrenado"""
        print(f"\nProbando agente entrenado por {episodes} episodios...")
        
        test_rewards = []
        test_steps = []
        
        for episode in range(episodes):
            if render:
                env = gym.make('MountainCar-v0', render_mode='human')
            else:
                env = self.env
                
            state, _ = env.reset()
            state = self.discretize_state(state)
            
            total_reward = 0
            steps = 0
            
            # Usar política greedy (sin exploración)
            for step in range(200):
                action = np.argmax(self.q_table[state])  # Política greedy
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.discretize_state(next_state)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            test_rewards.append(total_reward)
            test_steps.append(steps)
            
            if render:
                print(f"Episodio de prueba {episode+1}: {steps} pasos, reward: {total_reward}")
                env.close()
        
        print(f"\nResultados de prueba:")
        print(f"Pasos promedio: {np.mean(test_steps):.1f}")
        print(f"Reward promedio: {np.mean(test_rewards):.2f}")
    
    def plot_results(self):
        """Graficar resultados del entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Rewards por episodio
        ax1.plot(self.rewards, alpha=0.6)
        # Media móvil
        window = 50
        if len(self.rewards) >= window:
            moving_avg = np.convolve(self.rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(self.rewards)), moving_avg, 'r-', linewidth=2)
        ax1.set_title('Reward por Episodio')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Reward Total')
        ax1.grid(True, alpha=0.3)
        
        # Pasos por episodio
        ax2.plot(self.episode_lengths, alpha=0.6)
        if len(self.episode_lengths) >= window:
            moving_avg = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(self.episode_lengths)), moving_avg, 'r-', linewidth=2)
        ax2.set_title('Pasos por Episodio')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Número de Pasos')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    # Crear entorno
    env = gym.make('MountainCar-v0')
    
    # Crear agente Q-Learning
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.95)
    
    print("Iniciando entrenamiento con Q-LEARNING...")
    print("Algoritmo: Aprende la política ÓPTIMA independiente de la exploración")
    
    # Entrenar
    agent.train(episodes=1000)
    
    # Mostrar resultados
    agent.plot_results()
    
    # Probar agente
    agent.test_agent(episodes=3, render=True)
    
    env.close()

if __name__ == "__main__":
    main()