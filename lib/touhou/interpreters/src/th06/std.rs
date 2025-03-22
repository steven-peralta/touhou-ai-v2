//! Interpreter of STD files.

use touhou_formats::th06::std::{Stage, Call, Instruction};
use crate::th06::interpolator::{Interpolator3, Formula};
use touhou_utils::math::{Mat4, setup_camera};
use std::cell::RefCell;
use std::rc::Rc;

/// Interpreter for Stage.
pub struct StageRunner {
    /// XXX: no pub.
    pub stage: Rc<RefCell<Stage>>,
    frame: u32,

    position: Interpolator3<f32>,
    direction: Interpolator3<f32>,

    /// XXX: no pub.
    pub fog_color: [f32; 4],
    /// XXX: no pub.
    pub fog_near: f32,
    /// XXX: no pub.
    pub fog_far: f32,
}

impl StageRunner {
    /// Create a new StageRunner attached to a Stage.
    pub fn new(stage: Rc<RefCell<Stage>>) -> StageRunner {
        StageRunner {
            stage,
            frame: 0,
            position: Interpolator3::new([0., 0., 0.], 0, [0., 0., 0.], 0, Formula::Linear),
            direction: Interpolator3::new([0., 0., 0.], 0, [0., 0., 0.], 0, Formula::Linear),
            fog_color: [1.; 4],
            fog_near: 0.,
            fog_far: 1000.,
        }
    }

    /// Advance the simulation one frame.
    pub fn run_frame(&mut self) {
        let stage = self.stage.borrow();

        for Call { time, instr } in stage.script.iter() {
            let time = *time;
            if time != self.frame {
                continue;
            }

            println!("{} {:?}", time, instr);

            match *instr {
                Instruction::SetViewpos(x, y, z) => {
                    self.position.set_start(time, [x, y, z]);
                    for Call { time, instr } in stage.script.iter().cloned() {
                        if time <= self.frame {
                            continue;
                        }
                        if let Instruction::SetViewpos(x, y, z) = instr {
                            self.position.set_end(time, [x, y, z]);
                            break;
                        }
                    }
                }
                Instruction::SetFog(b, g, r, a, near, far) => {
                    self.fog_color = [r as f32 / 255., g as f32 / 255., b as f32 / 255., a as f32 / 255.];
                    self.fog_near = near;
                    self.fog_far = far;
                }
                Instruction::SetViewpos2(dx, dy, dz) => {
                    let direction = [dx, dy, dz];
                    self.direction.set_start(time, if time == 0 { direction } else { self.direction.values(time) });
                    self.direction.set_end_values(direction);
                }
                Instruction::StartInterpolatingViewpos2(frame, _, _) => {
                    self.direction.set_end_frame(time + frame);
                }
                Instruction::StartInterpolatingFog(frame, _, _) => {
                }
                Instruction::Unknown(_, _, _) => {
                }
            }
        }

        self.frame += 1;
    }

    /// Generate the model-view matrix for the current frame.
    pub fn get_model_view(&self) -> Mat4 {
        let [x, y, z] = self.position.values(self.frame);

        let [dx, dy, dz] = self.direction.values(self.frame);

        let view = setup_camera(dx, dy, dz);

        let model = Mat4::new([[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [-x, -y, -z, 1.]]);
        model * view
    }
}
