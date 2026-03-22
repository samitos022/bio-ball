import numpy as np
import scipy.special
import config_gecco as cfg

def generate_xt_map(grid_res=cfg.GRID_RES, phase="Attacking possession"):
    x = np.linspace(0, 1, grid_res)
    y = np.linspace(0, 1, grid_res)
    X, Y = np.meshgrid(x, y)
    
    if "Attacking" in phase or phase == "pa":
        x_dist = 1.0 - X
    else:
        x_dist = X 
        
    y_dist = np.abs(Y - 0.5)
        
    # 1. Base xT: decadimento verso le fasce
    base_xt = np.exp(-(2.5 * x_dist + 3.0 * (y_dist**2)))
    
    # 2. Creiamo due "dossi" gaussiani nei mezzi spazi (y approx 0.25 e 0.75) nella trequarti offensiva
    half_space_bonus = 0.3 * np.exp(-(5.0 * x_dist + 40.0 * (Y - 0.25)**2)) + \
                       0.3 * np.exp(-(5.0 * x_dist + 40.0 * (Y - 0.75)**2))
                       
    # 3. Un piccolo incentivo a stare larghi (y vicino a 0 o 1) per "strecciare" la difesa
    flank_bonus = 0.15 * (Y**4 + (1-Y)**4) * (1.0 - x_dist)
    xt_map = base_xt + half_space_bonus + flank_bonus
    
    return xt_map / np.max(xt_map)

def calculate_pitch_control(home_coords, away_coords, ball_pos, grid_res=cfg.GRID_RES):

    # 1. Creazione della griglia (Shape: grid_res, grid_res)
    x_lin = np.linspace(0, 1, grid_res)
    y_lin = np.linspace(0, 1, grid_res)
    X, Y = np.meshgrid(x_lin, y_lin)
    
    def get_team_influence(coords):
        if len(coords) == 0:
            return np.zeros((grid_res, grid_res))
        
        p_x = coords[:, 0].reshape(-1, 1, 1)
        p_y = coords[:, 1].reshape(-1, 1, 1)
        
        dist_to_ball = np.linalg.norm(coords - ball_pos, axis=1).reshape(-1, 1, 1)
        
        dynamic_radii = cfg.PLAYER_INFLUENCE_R + (0.15 * dist_to_ball)
        R_sq = dynamic_radii ** 2
        
        distance_sq = (X - p_x)**2 + (Y - p_y)**2
        
        influence = np.sum(np.exp(-distance_sq / (2 * R_sq)), axis=0)
        return influence

    # 2. Calcolo influenzE
    influence_home = get_team_influence(home_coords)
    influence_away = get_team_influence(away_coords)
    
    # 3. Pitch Control Sigmoide
    pc_home = scipy.special.expit(influence_home - influence_away)
    
    return pc_home