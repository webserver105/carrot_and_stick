# 🤖 Carrot & Stick
Welcome to the Electronics and Robotics Club (ERC) Reinforcement Learning Workshop.

Today, we aren't just running pre-written code and watching a screen. We are building the "brain" of an industrial robotic arm from scratch. You will take a 6-DOF UR5 arm and teach it how to see, think, and grasp a target using Soft Actor-Critic (SAC) Reinforcement Learning.

Your objective? Write the mathematical reward function that shapes the AI's psychology, forcing it to learn how to reach the blue cube as efficiently as possible, train it, and then test it live.

Let's power up the environment.
## ⚙️ Phase 1: Environment Setup
Physics engines like PyBullet are heavy. To ensure your laptop doesn't crash trying to compile C++ from scratch, you must use Python 3.10 for this environment.

### **Step 1: Clone the Repository**

Open your terminal or command prompt and pull down the workshop files:
```
git clone https://github.com/webserver105/carrot_and_stick.git
cd carrot_and_stick
```
### **Step 2: Create a Dedicated 3.10 Virtual Environment**

We need an isolated space.
Do not use Python 3.11, 3.12, or 3.13, or the physics engine will fail to install.

**For Windows:**
```
py -3.10 -m venv venv
venv\Scripts\activate
```
*(If py is not recognized, use the full path to your Python 3.10 executable, e.g., C:\path\to\python3.10.exe -m venv venv)*

**For MAC/ Linux**
```
python3.10 -m venv venv
source venv/bin/activate
```
### **Step 3: Install the Neural Stack**

Upgrade your package manager and install the exact pre-compiled libraries required for the physics engine and the AI:

```
python -m pip install --upgrade pip
pip install stable-baselines3 pybullet gymnasium numpy
```


## 🧠 Phase 2: The Challenge
Right now, the AI has a body, but it doesn't know why it is moving. Before we can train it, you must give it a purpose by writing its Reward Function.

1. Open **ur5_env.py** in your code editor.
2. Scroll down to the step() function.
3. Find the block labeled 🎯 ERC WORKSHOP CHALLENGE: THE DENSE REWARD.
4. You are provided the variable distance_to_target (the Euclidean distance between the robot's gripper and the cube).
5. **The Challenge:** The AI wants to maximize its total score. Write a mathematical formula using distance_to_target that creates a penalty forcing the arm to move closer to the cube on every single frame.

*(Hint: If you make the reward positive, the AI will realize that moving AWAY from the cube gives it infinite points!)*

## 🚀 Phase 3: Training & Testing the Model
Once you have written your custom reward function, it's time to let the neural network learn from it.

### Run the script:
```
python main_rl.py
```
The simulation will teleport the box randomly around the table, forcing the AI to learn how to reach every possible angle using the math you just wrote!

1. Stop the training script (Ctrl+C in the terminal)
2. In **main_rl.py**, comment out training and turn on testing:
   ```
   def main():
    #train model
    # train_algo()  
    #test model
    test_algo()
   ```
3. Run the script again
   ```
   python main_rl.py
   ```

**How to use the Interactive GUI:**
*  The 3D PyBullet window will open.
*  Look at the right side of the screen under the "Params" tab.
*  You will see sliders for Target X and Target Y.
*  Adjust the sliders to pick a coordinate, then click the "Place Box & Start" button.
*  The blue cube will spawn at your exact coordinates. Watch your trained AI sweep in and grab it!
