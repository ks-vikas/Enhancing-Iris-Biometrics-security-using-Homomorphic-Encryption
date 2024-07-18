![](contraLogo.jpeg)

# COP 701 ASSIGNMENT#2 - Contra Game 

This is an assignment for course "Software Systems LAB" conducted by Dr. Smruti Ranjan Sarangi (Sem 1, 2023, IIT Delhi).  

Submitted by: Bhuvnesh Kumar (2023MCS2011) and Vikas Kumar Saini (2023MCS2492) 

### **Problem Statement** 

Implement the Contra game using Unity3D or the Unreal engine. 

1. Make it a standalone application. 
1. Incorporate very nice visual elements. 
1. Make the graphics very smooth. 
1. Gradually make a level harder. 
1. Have at least three levels. Be creative and improvise. Add at least two new features. 

### **Game details** 

- Implemented the Contra game using Unity3D engine as a standalone application
- Nice Visual Elements and smooth graphics were incorporated
- Levels are made harder by varying the number of traps and enemies, and by incorporating level designs of various difficulty levels
- Three levels are designed in the game. Various creative features are added throughout the game. Pause/ Resume, coins collect, player hold rifle, shoot in all direction 
 and pass through platform on crouch.

#### 1. **Game Overview** 

**”Contra”** is a classic run-and-gun platformer game known for its intense action and challenging levels. Players control soldiers fighting against alien and enemy forces in a 2D side-scrolling environment. 

#### 2. **Game mechanics** 

The following features are included: 

- Player Controls: Player can move left/right, jump, crouch, and shoot
- Health: Player have a health bar; getting hit reduces health
- Enemies: Enemies include alien creatures, soldiers, and mechanical foes 
- Display: The game screen should display health, current weapon, and score
#### 3. **Level design** 
- Themes: Levels include jungle, ice, and Fortress themes
- Obstacles: Levels feature platforms, traps, and moving hazards 
- Boss Battles: Each level ends with a challenging boss battle 
#### 4. **Bonus** 

The following features are added for extra credit: 

- **Power-ups**: Collectible power-ups for **Changing bullets** (**Spread shot, Burst shot**) 
- **Power-ups**: Collectible power-ups **for Changing Guns** (**Flame Thrower, Shot Gun**)
- **Lives**: Player has a limited number of lives; extra lives can be earned if lives are less than max possible lives
- **Health**: Player has a limited health; additional health can be earned when it reduces
- **Coins**: Player collects coins to increase the score
- **Moving Platforms**: Player has to jump on platform to cross the area
- **Moving Hazards**: Moving enemies, Saws 
- Platform Crossing: Player can **jump to upper platform** without colliding
- Platform Crossing: Player **can go to lower platform** without colliding 
- **Boss sends enemies**: While fighting with boss along with shooting fireballs, boss also sends enemies
- **Enemies take some of player’s health** on death if player is near enemy when it exploded 
- **Enemies on moving platform** 
- **Health and lives on moving platform** 
- **Traps**: Various traps are installed at different locations. These include **Spikes and rotating saws** 
- **Menus: Start menu, Pause Menu, Game Over Menu, load levels option menu** are also provided for in game transitions 
- **Mechanical foes**: Turrets are installed
- **Respawn**: Player respawns after health is zero
- **Gun Display**: Gun is displayed in players hand

### **Scripts** 
#### 1. **Enemy Scripts** 
- **EnemyBullet Script:** Tracks player current position then sends the bullet at the player’s position and if player is hit the bullet is destroyed
- **EnemyFacePlayer Script**: If checks player position with respect to enemy, then if enemy is not facing player then rotates it to face player
- **EnemyFire Script:** Checks if player is in range. Then shoots bullets at intervals at the player
- **EnemyLife Script:** Checks Enemy life, if life is less than zero then changes enemy animation to dying animation. Also has a function to destroy the enemy object which is called from the event called at the die animation of enemy.  

- **EnemyRotation:** Turret rotation scripts. Rotates turret in player’s direction
- **EnemyRunLeft Script:** Assigns the direction of  movement of enemy to the left
- **SpawnerEnemy Script:** Spawns the enemies at the spawner position. It holds enemies in an array. Then spawns a number of enemies  at random time intervals. It also sets the enemy movement speed at random at time of spawning the enemy
#### 2. **Health Scripts** 
- **Healthbar Script:** Fills the player’s health bar on canvas
- **Lifebar Script:** Fills the player’s life bar on canvas
#### 3. **MovingPlatform Scripts** 
- **StickToPlatform Script:** Makes the player move with platform if it is standing on it
- **WaypointFollower Script:** Used to move platform/objects between the two points
#### 4. **Player Scripts** 
- **BulletScript Script:** If player bullet hits enemy then decrease the enemy health life, and destroys bullet
- **ChangeRifle Script:** Changes the player rifle when player takes the rifle powerups
- **displayScore Script:** Displays the Current score on the canvas and update the score if coin is collected
- **FireBullet Script:** Fires the bullet from the firepoint depending on the powerup of the bullet player has taken
- **MainCameraController Script:** camera follows the player
- **PlayerLife Script:** Controls player life. Updates when player takes damage  or powerup
- **PlayerMovement Script:** Controls the player movements and animations when keyboard movement keys are pressed
- **Powerup Script:** Changes bullets when bullet power up is taken
- **RifleRotation Script:** Rotates Player rifles with mouse position
- **ScoreCalculate:** Calculates and updates the score when player bullet hits enemy
#### 5. **Scenes Scripts** 
- **PauseMenu Script:** Pauses/resumes the gameplay, load other scenes
- **StartMenu Script:** Plays different levels, and quit applications
#### 6. **autoDestroy Script:** 
- Destroys the bullets which did not hit the player. Also destroys the distant past enemies which were not  killed by the player.

**NOTE: All graphical images are picked up from the internet. All levels’ objects  are designed by us using the those images.** 

### **Player Controls** 

- Move Left: A / LEFT ARROW KEY 
- Move Right: D / RIGHT ARROW KEY 
- Jump: SPACE 
- Crouch: S / Down arrow key 
- Aim: MOUSE
- Go to Lower terrain: double press S key 
- Single Shoot mode: LEFT CLICK / left Ctrl 
- Automatic Shoot mode : RIGHT CLICK / left Alt 
- Pause Game: P Key 

***\*Playing with Mouse is recommended instead of Touchpad.***  
***\*Best Game Play can be experienced in 1920x1080 Resolution.***

Best of Luck!!
