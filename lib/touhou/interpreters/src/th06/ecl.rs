//! ECL runner.

use touhou_formats::th06::ecl::{Ecl, SubInstruction};
use crate::th06::enemy::{Enemy, Offset, BulletAttributes, Position};
use touhou_utils::prng::Prng;
use std::cell::RefCell;
use std::rc::Rc;

macro_rules! gen_SetBulletAttributes {
    ($self:ident, $opcode:tt, $anim:ident, $sprite_index_offset:ident, $bullets_per_shot:ident,
     $number_of_shots:ident, $speed:ident, $speed2:ident, $launch_angle:ident, $angle:ident,
     $flags:ident) => {{
        let sprite_index_offset = $self.get_i32($sprite_index_offset as i32) as i16;
        let bullets_per_shot = $self.get_i32($bullets_per_shot) as i16;
        let number_of_shots = $self.get_i32($number_of_shots) as i16;
        let speed = $self.get_f32($speed);
        let speed2 = $self.get_f32($speed2);
        let launch_angle = $self.get_f32($launch_angle);
        let angle = $self.get_f32($angle);

        let mut enemy = $self.enemy.borrow_mut();
        enemy.set_bullet_attributes($opcode, $anim, sprite_index_offset, bullets_per_shot,
                                    number_of_shots, speed, speed2, launch_angle, angle, $flags);
    }};
}

#[derive(Clone, Default)]
struct StackFrame {
    frame: i32,
    ip: i32,
    //ins122_callback: Option<Box<FnMut(Enemy)>>,
    ints1: [i32; 4],
    floats: [f32; 4],
    ints2: [i32; 4],
    comparison_reg: i32,
    sub: u16,
}

/// Interpreter for enemy scripts.
#[derive(Default)]
pub struct EclRunner {
    /// XXX
    pub enemy: Rc<RefCell<Enemy>>,

    ecl: Option<Ecl>,
    /// XXX
    pub running: bool,
    frame: StackFrame,
    // TODO: there are only 8 of these.
    stack: Vec<StackFrame>,
}

impl EclRunner {
    /// Create a new ECL runner.
    pub fn new(ecl: &Ecl, enemy: Rc<RefCell<Enemy>>, sub: u16) -> EclRunner {
        let mut ecl_runner = EclRunner {
            enemy,
            // XXX: no clone.
            ecl: Some(ecl.clone()),
            running: true,
            ..Default::default()
        };
        ecl_runner.frame.sub = sub;
        ecl_runner
    }

    /// Advance the ECL of a single frame.
    pub fn run_frame(&mut self) {
        while self.running {
            let ecl = self.ecl.clone().unwrap();
            let sub = &ecl.subs[self.frame.sub as usize];
            let call = match sub.instructions.get(self.frame.ip as usize) {
                Some(call) => call,
                None => {
                    self.running = false;
                    break;
                }
            };

            if call.time > self.frame.frame {
                break;
            }
            self.frame.ip += 1;

            let rank = self.enemy.borrow().get_rank();
            if (call.rank_mask & rank).is_empty() {
                continue;
            }

            if call.time == self.frame.frame {
                self.run_instruction(call.instr.clone());
            }
        }
        self.frame.frame += 1;
    }

    fn get_i32(&self, var: i32) -> i32 {
        let enemy = self.enemy.borrow();
        match var {
            -10001 => self.frame.ints1[0],
            -10002 => self.frame.ints1[1],
            -10003 => self.frame.ints1[2],
            -10004 => self.frame.ints1[3],
            -10005 => self.frame.floats[0] as i32,
            -10006 => self.frame.floats[1] as i32,
            -10007 => self.frame.floats[2] as i32,
            -10008 => self.frame.floats[3] as i32,
            -10009 => self.frame.ints2[0],
            -10010 => self.frame.ints2[1],
            -10011 => self.frame.ints2[2],
            -10012 => self.frame.ints2[3],
            -10013 => enemy.get_rank().bits() as i32,
            -10014 => enemy.get_difficulty(),
            -10015 => enemy.pos.x as i32,
            -10016 => enemy.pos.y as i32,
            -10017 => enemy.z as i32,
            -10018 => unimplemented!(),
            -10019 => unimplemented!(),
            -10020 => unreachable!(),
            -10021 => unimplemented!(),
            -10022 => enemy.frame as i32,
            -10023 => unreachable!(),
            -10024 => enemy.life as i32,
            -10025 => unimplemented!(),
            _ => var
        }
    }

    fn get_f32(&self, var: f32) -> f32 {
        let enemy = self.enemy.borrow();
        match var {
            -10001.0 => self.frame.ints1[0] as f32,
            -10002.0 => self.frame.ints1[1] as f32,
            -10003.0 => self.frame.ints1[2] as f32,
            -10004.0 => self.frame.ints1[3] as f32,
            -10005.0 => self.frame.floats[0],
            -10006.0 => self.frame.floats[1],
            -10007.0 => self.frame.floats[2],
            -10008.0 => self.frame.floats[3],
            -10009.0 => self.frame.ints2[0] as f32,
            -10010.0 => self.frame.ints2[1] as f32,
            -10011.0 => self.frame.ints2[2] as f32,
            -10012.0 => self.frame.ints2[3] as f32,
            -10013.0 => enemy.get_rank().bits() as f32,
            -10014.0 => enemy.get_difficulty() as f32,
            -10015.0 => enemy.pos.x,
            -10016.0 => enemy.pos.y,
            -10017.0 => enemy.z,
            -10018.0 => unimplemented!(),
            -10019.0 => unimplemented!(),
            -10020.0 => unreachable!(),
            -10021.0 => unimplemented!(),
            -10022.0 => enemy.frame as f32,
            -10023.0 => unreachable!(),
            -10024.0 => enemy.life as f32,
            -10025.0 => unimplemented!(),
            _ => var
        }
    }

    fn set_i32(&mut self, var: i32, value: i32) {
        let mut enemy = self.enemy.borrow_mut();
        match var {
            -10001 => self.frame.ints1[0] = value,
            -10002 => self.frame.ints1[1] = value,
            -10003 => self.frame.ints1[2] = value,
            -10004 => self.frame.ints1[3] = value,
            -10005 => unimplemented!(),
            -10006 => unimplemented!(),
            -10007 => unimplemented!(),
            -10008 => unimplemented!(),
            -10009 => self.frame.ints2[0] = value,
            -10010 => self.frame.ints2[1] = value,
            -10011 => self.frame.ints2[2] = value,
            -10012 => self.frame.ints2[3] = value,
            -10013 => unreachable!(),
            -10014 => unreachable!(),
            -10015 => unimplemented!(),
            -10016 => unimplemented!(),
            -10017 => unimplemented!(),
            -10018 => unreachable!(),
            -10019 => unreachable!(),
            -10020 => unreachable!(),
            -10021 => unreachable!(),
            -10022 => enemy.frame = value as u32,
            -10023 => unreachable!(),
            -10024 => enemy.life = value as u32,
            -10025 => unreachable!(),
            _ => panic!("Unknown variable {}", var)
        }
    }

    fn set_f32(&mut self, var: f32, value: f32) {
        let mut enemy = self.enemy.borrow_mut();
        match var {
            -10001.0 => unimplemented!(),
            -10002.0 => unimplemented!(),
            -10003.0 => unimplemented!(),
            -10004.0 => unimplemented!(),
            -10005.0 => self.frame.floats[0] = value,
            -10006.0 => self.frame.floats[1] = value,
            -10007.0 => self.frame.floats[2] = value,
            -10008.0 => self.frame.floats[3] = value,
            -10009.0 => unimplemented!(),
            -10010.0 => unimplemented!(),
            -10011.0 => unimplemented!(),
            -10012.0 => unimplemented!(),
            -10013.0 => unreachable!(),
            -10014.0 => unreachable!(),
            -10015.0 => enemy.pos.x = value,
            -10016.0 => enemy.pos.y = value,
            -10017.0 => enemy.z = value,
            -10018.0 => unreachable!(),
            -10019.0 => unreachable!(),
            -10020.0 => unreachable!(),
            -10021.0 => unreachable!(),
            -10022.0 => unimplemented!(),
            -10023.0 => unreachable!(),
            -10024.0 => unimplemented!(),
            -10025.0 => unreachable!(),
            _ => panic!("Unknown variable {}", var)
        }
    }

    fn get_prng(&mut self) -> Rc<RefCell<Prng>> {
        let enemy = self.enemy.borrow();
        enemy.prng.upgrade().unwrap()
    }

    fn run_instruction(&mut self, instruction: SubInstruction) {
        println!("Running instruction {:?}", instruction);
        match instruction {
            SubInstruction::Noop() => {
                // really
            }
            // 1
            SubInstruction::Destroy(_unused) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.removed = true;
            }
            // 2
            SubInstruction::RelativeJump(frame, ip) => {
                self.frame.frame = frame;
                // ip = ip + flag in th06
                self.frame.ip = ip;
                // we jump back to the main of the interpreter
            }
            // 3
            // GHIDRA SAYS THERE IS A COMPARISON_REG BUFFER BUT THERE IS NOT!!!
            //
            // MOV        ECX,dword ptr [EBP + 0x8]                     jumptable 00407544 case 31
            // CMP        dword ptr [0x9d4 + ECX],0x0
            // JLE        LAB_00407abb
            // aka ECX = enemy pointer
            // ECX->9d4 (aka enemy_pointer_copy->comparison_reg) == 0
            // only the pointer is copied, not the value, thus we are safe
            SubInstruction::RelativeJumpEx(frame, ip, var_id) => {
                // TODO: counter_value is a field of "enemy" in th06, to check
                let counter_value = self.get_i32(var_id) - 1;
                if counter_value > 0 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 4
            SubInstruction::SetInt(var_id, value) => {
                self.set_i32(var_id, value);
            }
            // 5
            SubInstruction::SetFloat(var_id, value) => {
                self.set_f32(var_id as f32, value);
            }
            // 6
            SubInstruction::SetRandomInt(var_id, maxval) => {
                let random = self.get_prng().borrow_mut().get_u32() as i32;
                self.set_i32(var_id, random % self.get_i32(maxval));
            }
            // 7
            SubInstruction::SetRandomIntMin(var_id, maxval, minval) => {
                let random = self.get_prng().borrow_mut().get_u32() as i32;
                self.set_i32(var_id, (random % self.get_i32(maxval)) + self.get_i32(minval));
            }
            // 8
            SubInstruction::SetRandomFloat(var_id, maxval) => {
                let random = self.get_prng().borrow_mut().get_f64() as f32;
                self.set_f32(var_id as f32, self.get_f32(maxval) * random)
            }
            // 9
            SubInstruction::SetRandomFloatMin(var_id, maxval, minval) => {
                let random = self.get_prng().borrow_mut().get_f64() as f32;
                self.set_f32(var_id as f32, self.get_f32(maxval) * random + self.get_f32(minval))
            }
            // 10
            SubInstruction::StoreX(var_id) => {
                let x = {
                    let enemy = self.enemy.borrow();
                    enemy.pos.x
                };
                // TODO: is this really an i32?
                self.set_i32(var_id, x as i32);
            }
            // 11
            SubInstruction::StoreY(var_id) => {
                let y = {
                    let enemy = self.enemy.borrow();
                    enemy.pos.y
                };
                self.set_i32(var_id, y as i32);
            }
            // 12
            SubInstruction::StoreZ(var_id) => {
                let z = {
                    let enemy = self.enemy.borrow();
                    enemy.z
                };
                self.set_i32(var_id, z as i32);
            }
            // 13(int), 20(float), same impl in th06
            SubInstruction::AddInt(var_id, a, b) => {
                self.set_i32(var_id, self.get_i32(a) + self.get_i32(b));
            }
            SubInstruction::AddFloat(var_id, a, b) => {
                self.set_f32(var_id as f32, self.get_f32(a) + self.get_f32(b));
            }
            // 14(int), 21(float), same impl in th06
            SubInstruction::SubstractInt(var_id, a, b) => {
                self.set_i32(var_id, self.get_i32(a) - self.get_i32(b));
            }
            SubInstruction::SubstractFloat(var_id, a, b) => {
                self.set_f32(var_id as f32, self.get_f32(a) - self.get_f32(b));
            }
            // 15(int), 22(unused)
            SubInstruction::MultiplyInt(var_id, a, b) => {
                self.set_i32(var_id, self.get_i32(a) * self.get_i32(b));
            }
            /*
            SubInstruction::MultiplyFloat(var_id, a, b) => {
                self.set_f32(var_id as f32, self.get_f32(a) * self.get_f32(b));
            }
            */
             // 16(int), 23(unused)
            SubInstruction::DivideInt(var_id, a, b) => {
                self.set_i32(var_id, self.get_i32(a) / self.get_i32(b));
            }

            SubInstruction::DivideFloat(var_id, a, b) => {
                self.set_f32(var_id as f32, self.get_f32(a) / self.get_f32(b));
            }

            // 17(int) 24(unused)
            SubInstruction::ModuloInt(var_id, a, b) => {
                self.set_i32(var_id, self.get_i32(a) % self.get_i32(b));
            }

            SubInstruction::ModuloFloat(var_id, a, b) => {
                self.set_f32(var_id as f32, self.get_f32(a) % self.get_f32(b));
            }

            // 18
            // setval used by pytouhou, but not in game(???)
            SubInstruction::Increment(var_id) => {
                self.set_i32(var_id, self.get_i32(var_id) + 1);
            }

            // 19
            SubInstruction::Decrement(var_id) => {
                self.set_i32(var_id, self.get_i32(var_id) - 1);
            }

            //25
            SubInstruction::GetDirection(var_id, x1, y1, x2, y2) => {
                //__ctrandisp2 in ghidra, let's assume from pytouhou it's atan2
                self.set_f32(var_id as f32, (self.get_f32(y2) - self.get_f32(y1)).atan2(self.get_f32(x2) - self.get_f32(x1)));
            }

            // 26
            SubInstruction::FloatToUnitCircle(var_id) => {
                // TODO: atan2(var_id, ??) is used by th06, maybe ?? is pi?
                // we suck at trigonometry so let's use pytouhou for now
                self.set_f32(var_id as f32, (self.get_f32(var_id as f32) + std::f32::consts::PI) % (2. * std::f32::consts::PI) - std::f32::consts::PI);
            }

            // 27(int), 28(float)
            SubInstruction::CompareInts(a, b) => {
                let a = self.get_i32(a);
                let b = self.get_i32(b);
                if a < b {
                    self.frame.comparison_reg = -1;
                }
                else if  a == b {
                    self.frame.comparison_reg = 0;
                }
                else {
                    self.frame.comparison_reg = 1;
                }
            }
            SubInstruction::CompareFloats(a, b) => {
                let a = self.get_f32(a);
                let b = self.get_f32(b);
                if a < b {
                    self.frame.comparison_reg = -1;
                }
                else if  a == b {
                    self.frame.comparison_reg = 0;
                }
                else {
                    self.frame.comparison_reg = 1;
                }
            }
            // 29
            SubInstruction::RelativeJumpIfLowerThan(frame, ip) => {
                if self.frame.comparison_reg == -1 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 30
            SubInstruction::RelativeJumpIfLowerOrEqual(frame, ip) => {
                if self.frame.comparison_reg != 1 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 31
            SubInstruction::RelativeJumpIfEqual(frame, ip) => {
                if self.frame.comparison_reg == 0 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 32
            SubInstruction::RelativeJumpIfGreaterThan(frame, ip) => {
                if self.frame.comparison_reg == 1 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 33
            SubInstruction::RelativeJumpIfGreaterOrEqual(frame, ip) => {
                if self.frame.comparison_reg != -1 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 34
            SubInstruction::RelativeJumpIfNotEqual(frame, ip) => {
                if self.frame.comparison_reg != 0 {
                    self.run_instruction(SubInstruction::RelativeJump(frame, ip));
                }
            }
            // 35
            SubInstruction::Call(sub, param1, param2) => {
                self.stack.push(self.frame.clone());
                self.frame.sub = sub as u16;
                self.frame.ints1[0] = param1;
                self.frame.floats[0] = param2;
                self.frame.frame = 0;
                self.frame.ip = 0;
            }

            // 36
            SubInstruction::Return() => {
                self.frame = self.stack.pop().unwrap();
            }
            // 37
            SubInstruction::CallIfSuperior(sub, param1, param2, a, b) => {
                if self.get_i32(a) < self.get_i32(b) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }
            // 38
            SubInstruction::CallIfSuperiorOrEqual(sub, param1, param2, a, b) => {
                if self.get_i32(a) <= self.get_i32(b) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }
            // 39
            SubInstruction::CallIfEqual(sub, param1, param2, a, b) => {
                if self.get_i32(a) == self.get_i32(b) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }
            // 40
            SubInstruction::CallIfInferior(sub, param1, param2, a, b) => {
                if self.get_i32(b) < self.get_i32(a) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }

            // 41
            SubInstruction::CallIfInferiorOrEqual(sub, param1, param2, a, b) => {
                if self.get_i32(b) <= self.get_i32(a) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }
            //42
            SubInstruction::CallIfNotEqual(sub, param1, param2, a, b) => {
                if self.get_i32(a) != self.get_i32(b) {
                    self.run_instruction(SubInstruction::Call(sub, param1, param2));
                }
            }

            // 43
            SubInstruction::SetPosition(x, y, z) => {
                let (x, y, z) = (self.get_f32(x), self.get_f32(y), self.get_f32(z));
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_pos(x, y, z);
            }
            // 44
            /*
            SubInstruction::SetAngularSpeed(x, y, z) => {
                // same as above, except for angular speed
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_angular_speed(self.get_f32(x), self.get_f32(y), self.get_f32(z));
            }
            */
            // 45
            SubInstruction::SetAngleAndSpeed(angle, speed) => {
                let angle = self.get_f32(angle);
                let speed = self.get_f32(speed);
                let mut enemy = self.enemy.borrow_mut();
                enemy.update_mode = 0;
                enemy.angle = angle;
                enemy.speed = speed;
            }
            // 46
            SubInstruction::SetRotationSpeed(speed) => {
                let rotation_speed = self.get_f32(speed);
                let mut enemy = self.enemy.borrow_mut();
                enemy.update_mode = 0;
                enemy.rotation_speed = rotation_speed;
            }
            // 47
            SubInstruction::SetSpeed(speed) => {
                let speed = self.get_f32(speed);
                let mut enemy = self.enemy.borrow_mut();
                enemy.update_mode = 0;
                enemy.speed = speed;
            }
            // 48
            SubInstruction::SetAcceleration(acceleration) => {
                let acceleration = self.get_f32(acceleration);
                let mut enemy = self.enemy.borrow_mut();
                enemy.update_mode = 0;
                enemy.acceleration = acceleration;
            }
            // 49
            SubInstruction::SetRandomAngle(min_angle, max_angle) => {
                let angle = self.get_prng().borrow_mut().get_f64() as f32 * (max_angle - min_angle) + min_angle;
                let mut enemy = self.enemy.borrow_mut();
                enemy.angle = angle;
            }
            // 51
            SubInstruction::TargetPlayer(delta_angle, speed) => {
                let speed = self.get_f32(speed);
                let mut enemy = self.enemy.borrow_mut();
                let game = enemy.game.upgrade().unwrap();
                let player = game.borrow().get_player();
                enemy.update_mode = 0;
                enemy.speed = speed;
                enemy.angle = enemy.get_angle_to(player) + delta_angle;
            }

            // 52 to 64 are different interlacing fields

            // 65
            // to note: in game a flag is set to enable the screenbox and is set by 66 to disable
            // it on top of setting our values. But we have a good engine and can detect if that's
            // changed without setting a flag :)
            SubInstruction::SetScreenBox(xmin, ymin, xmax, ymax) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.screen_box = Some((xmin, ymin, xmax, ymax));
            }
             // 66
            SubInstruction::ClearScreenBox() => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.screen_box = None;
            }

            // 67 to 75 are set bullet attributes and it seems a pain to reverse rn
            SubInstruction::SetBulletAttributes1(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 67, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes2(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 68, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes3(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 69, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes4(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 70, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes5(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 71, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes6(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 74, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }
            SubInstruction::SetBulletAttributes7(anim, sprite_index_offset, bullets_per_shot,
                                                 number_of_shots, speed, speed2, launch_angle,
                                                 angle, flags) => {
                gen_SetBulletAttributes!(self, 75, anim, sprite_index_offset, bullets_per_shot,
                                         number_of_shots, speed, speed2, launch_angle, angle,
                                         flags);
            }

            // 76
            SubInstruction::SetBulletInterval(interval) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_bullet_launch_interval(0, interval);
            }

            // 77
            SubInstruction::SetBulletIntervalEx(interval) => {
                let rand_start = self.get_prng().borrow_mut().get_u32();

                let mut enemy = self.enemy.borrow_mut();
                enemy.set_bullet_launch_interval(rand_start, interval);
            }

            // 78-79 are more interpolation flags
            // 78
            SubInstruction::DelayAttack() => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.delay_attack = true;
            }
            // 79
            SubInstruction::NoDelayAttack() => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.delay_attack = false;
            }
            // 80
            /*
            SubInstruction::NoClue() => {
                let mut enemy = self.enemy.borrow_mut();
                //bullet_pos = launch offset
                (enemy->bullet_attributes).bullets_per_shot = enemy.pos.x + enemy->bullet_pos.pos.x;
                (enemy->bullet_attributes).number_of_shots = enemy.pos.pos.y + enemy.bullet_pos.pos.y;
                (enemy->bullet_attributes).speed = enemy.z + bullet_pos.z;
                enemy.fire(bullet_attributes=bullet_attributes)
            }
            */

            // 81
            SubInstruction::SetBulletLaunchOffset(dx, dy, dz) => {
                let (dx, dy, dz) = (self.get_f32(dx), self.get_f32(dy), self.get_f32(dz));
                let mut enemy = self.enemy.borrow_mut();
                enemy.bullet_offset = Offset { dx, dy };
            }

            // 82
            SubInstruction::SetExtendedBulletAttributes(a, b, c, d, e, f, g, h) => {
                let (a, b, c, d) = (self.get_i32(a), self.get_i32(b), self.get_i32(c), self.get_i32(d));
                let (e, f, g, h) = (self.get_f32(e), self.get_f32(f), self.get_f32(g), self.get_f32(h));
                let mut enemy = self.enemy.borrow_mut();
                enemy.bullet_attributes.extended_attributes = (a, b, c, d, e, f, g, h);
            }

            // 83
            /*
            SubInstruction::ChangeBulletsIntoStarBonus() => {
                let mut game = self.game.borrow_mut();
                game.change_bullets_into_star_items();
            }
            */

            // 84
            SubInstruction::SetBulletSound(sound) => {
                let mut enemy = self.enemy.borrow_mut();
                if sound < 0 {
                    enemy.bullet_attributes.sound = None;
                } else {
                    // This assert isn’t part of the original engine, but it would crash on high
                    // values anyway.
                    assert!(sound <= 255);
                    enemy.bullet_attributes.sound = Some(sound as u8);
                }
            }

            // 85-86 ire newlaser functions

            // 87
            SubInstruction::SetUpcomingLaserId(laser_id) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.current_laser_id = laser_id;
            }

            // 88
            /*
            SubInstruction::AlterLaserAngle(laser_id, delta) => {
                let mut enemy = self.enemy.borrow_mut();
                if enemy.laser_by_id.contains_key(&laser_id) {
                    let mut laser = enemy.laser_by_id.get(&laser_id);
                    laser.angle += self.get_f32(delta);
                }
            }
            */

            // 89
            /*
            SubInstruction::AlterLaserAnglePlayer(laser_id, delta) => {
                let mut enemy = self.enemy.borrow_mut();
                if enemy.laser_by_id.contains_key(&laser_id) {
                    let mut laser = enemy.laser_by_id.get(laser_id);
                    let player = enemy.select_player();
                    laser.angle = enemy.get_angle(player) + angle;
                }
            }
            */

            // 90
            /*
            SubInstruction::RepositionLaser(laser_id, ox, oy, oz) => {
                let mut enemy = self.enemy.borrow_mut();
                if enemy.laser_by_id.contains_key(&laser_id) {
                    let mut laser = enemy.laser_by_id.get(&laser_id);
                    laser.set_base_pos(enemy.pos.x + ox, enemy.pos.y + oy, enemy.z + oz)
                }
            }
            */
            // 91
            // wat
            SubInstruction::LaserSetCompare(laser_id) => {
                let enemy = self.enemy.borrow_mut();
                // in game it checks if either the laser exists OR if one of its member is set to 0
                // which, uhhhh, we are not going to reimplement for obvious reasons
                // the correct implementation would be: if this laser does not exist have a
                // 1/100000 chance to continue, otherwise crash
                if enemy.laser_by_id.contains_key(&laser_id) {
                    // let's assume we gud
                    self.frame.comparison_reg = 1;
                }
                else{
                    self.frame.comparison_reg = 0;
                }
            }

            // 92
            /*
            SubInstruction::RepositionLaser(laser_id, ox, oy, oz) => {
                let mut enemy = self.enemy.borrow_mut();
                if enemy.laser_by_id.contains_key(&laser_id) {
                    let mut laser = enemy.laser_by_id.get(laser_id);
                    laser.cancel();
                }
            }
            */
            // 93
            // TODO: actually implement that hell
            SubInstruction::SetSpellcard(face, number, name) => {
                unimplemented!("spellcard start");

            }
            // 94
            SubInstruction::EndSpellcard() => {
                unimplemented!("spellcard end");

            }

            // 95
            SubInstruction::SpawnEnemy(sub, x, y, z, life, bonus, score) => {
                let x = self.get_f32(x);
                let y = self.get_f32(y);
                let _z = self.get_f32(z);
                let enemy = self.enemy.borrow_mut();
                let anm0 = enemy.anm0.upgrade().unwrap();
                let game = enemy.game.upgrade().unwrap();
                let enemy = Enemy::new(Position::new(x, y), life, bonus, score as u32, false, Rc::downgrade(&anm0), Rc::downgrade(&game));
                let ecl = self.ecl.clone().unwrap();
                let mut runner = EclRunner::new(&ecl, enemy, sub as u16);
                runner.run_frame();
            }

            // 96
            /*
            SubInstruction::KillEnemies() => {
                let mut game = self.game.borrow_mut();
                game.kill_enemies();
            }
            */



            // 97
            SubInstruction::SetAnim(index) => {
                // seems correct, game internally gets base_addr =(iVar13 + 0x1c934), pointer_addr = iVar14 * 4
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_anim(index as u8);
            }
            // 98
            SubInstruction::SetMultipleAnims(default, end_left, end_right, left, right, _unused) => {
                // _unused was supposed to set movement_dependant_sprites, but internally the game
                // assigns it 0xff
                // TODO: THIS DOES NOT CALL set_anim. this only assigns all parameters to their
                // internal struct. To check if the anims are set somewhere else
                let mut enemy = self.enemy.borrow_mut();
                enemy.movement_dependant_sprites = if left == -1 {
                    None
                } else {
                    enemy.set_anim(default as u8);
                    Some((end_left as u8, end_right as u8, left as u8, right as u8))
                };
            }
            // 99
            SubInstruction::SetAuxAnm(number, script) => {
                assert!(number < 8);
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_aux_anm(number, script);
            }

            // 100
            SubInstruction::SetDeathAnim(index) => {
                // TODO: takes 3 parameters in game as u8 unlike our single u32.
                // To reverse!
                let mut enemy = self.enemy.borrow_mut();
                enemy.death_anim = index;
            }
            // 101
            SubInstruction::SetBossMode(value) => {
                let enemy = self.enemy.borrow_mut();
                if value < 0 {
                    enemy.set_boss(false);
                }
                else {
                    // the boss pointer is written somewhere in memory and overwrote by a 0 when
                    // the boss mode is false, might want to look into that
                    enemy.set_boss(true);
                }
            }

            // 102
            // TODO: title says it all
            /*
            SubInstruction::ParticlesVoodooMagic(unk1, unk2, unk3, unk4, unk5) => {
            }
            */

            // 103
            SubInstruction::SetHitbox(width, height, depth) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.set_hitbox(width, height);
            }

            // 104
            SubInstruction::SetCollidable(collidable) => {
                // TODO: me and my siblings(105, 107, 117) are implemented as a single variable in the touhou 6
                // original engine. While our behaviour seems correct we might want to implement
                // that as a single variable
                // TODO[2]: THE BITFLAG MIGHT BE INCORRECT FOR OTHER SIBLING INSTRUCTIONS, the
                // behavior was DEFINITELY incorrect in pytouhou for SetTouchable at the very least
                let mut enemy = self.enemy.borrow_mut();
                enemy.collidable = (collidable&1) != 0;
            }

            // 105
            SubInstruction::SetDamageable(damageable) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.damageable = (damageable&1) != 0;
            }

            // 106
            SubInstruction::PlaySound(index) => {
                let enemy = self.enemy.borrow_mut();
                enemy.play_sound(index);
            }

            // 107
            SubInstruction::SetDeathFlags(death_flags) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.death_flags = death_flags;
            }
            // 108
            SubInstruction::SetDeathCallback(sub) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.death_callback = Some(sub);
            }

            // 109
            SubInstruction::MemoryWriteInt(value, index) => {
                unimplemented!("not again that damn foe corrupted my ret\\x41\\x41\\x41\\x41");
            }

            // 110
            /*
            SubInstruction::KillEnemy(enemy) => {
                let mut game = self.game.borrow_mut();
                game.kill_enemy(enemy);
            }
            */

            // 111
            /*
            SubInstruction::SetLife(value) => {
                let mut enemy = self.enemy.borrow_mut();
                let mut game = self.game.borrow_mut();
                enemy.life = value;
                game.interface.set_boss_life();
            }
            */
            // 112
            SubInstruction::SetElapsedTime(value) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.frame = value as u32;
            }
            // 113
            /*
            SubInstruction::SetLowLifeTrigger(value) => {
                let mut enemy = self.enemy.borrow_mut();
                let mut game = self.game.borrow_mut();
                enemy.low_life_trigger = value;
                game.interface.set_spell_life();
            }
            */
            // 114
            /*
             SubInstruction::SetLowLifeCallback(sub) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.low_life_callback.enable(self.switch_to_sub, (sub,));
            }
            */
            // 115
            /*
            SubInstruction::SetTimeout(timeout) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.frame = value;
                enemy.timeout = timeout;
            }
            */
            // 116
            /*
             SubInstruction::SetTimeoutCallback(sub) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.timeout_callback.enable(self.switch_to_sub, (sub,));
            }
            */


            // 117
            SubInstruction::SetTouchable(touchable) => {
                let mut enemy = self.enemy.borrow_mut();
                enemy.touchable = touchable != 0;
            }

            // 121
            // Here lies the Di Sword of sadness
            SubInstruction::CallSpecialFunction(function, arg) => {
                match function {
                    0 => {
                        let mut enemy = self.enemy.borrow_mut();
                        let game = enemy.game.upgrade().unwrap();
                        let mut game = game.borrow_mut();
                        //game.drop_particle(12, enemy.pos, 1, 0xffffffff);
                        //game.iter_bullets(|mut bullet| {
                        for bullet in game.bullets.iter() {
                            //game.new_effect(bullet.sprite, TODO);
                            let mut bullet = bullet.borrow_mut();
                            if arg == 0 {
                                bullet.speed = 0.;
                                bullet.dpos = [0., 0., 0.];
                            } else if arg == 1 {
                                bullet.flags |= 0x10;
                                bullet.frame = 220;
                                let rand_angle = game.prng.borrow_mut().get_f64() * 2. * std::f64::consts::PI - std::f64::consts::PI;
                                bullet.attributes[0] = (rand_angle.cos() * 0.01) as f32;
                                bullet.attributes[1] = (rand_angle.sin() * 0.01) as f32;
                            }
                        }
                    }
                    1 => {
                        let range_x = arg as f64;
                        let range_y = (arg as f32 * 0.75) as f64;
                        let rand_x = self.get_prng().borrow_mut().get_f64();
                        let rand_y = self.get_prng().borrow_mut().get_f64();
                        let mut enemy = self.enemy.borrow_mut();
                        let pos = [rand_x * range_x + enemy.pos.x as f64 - range_x / 2.,
                                   rand_y * range_y + enemy.pos.x as f64 - range_y / 2.];
                        enemy.bullet_attributes.fire();
                    }
                    3 => { // Patchouli’s dual sign spellcard selector
                        let mut enemy = self.enemy.borrow_mut();
                        let mut knowledge: [[i32; 3]; 4] =
                            [[0, 3, 1],
                             [2, 3, 4],
                             [1, 4, 0],
                             [4, 2, 3]];

                        //TODO: implement select_player and replace character by the correct one
                        //let character = enemy.select_player().character;
                        let character = 0;
                        for i in 1..=3 {
                            self.frame.ints1[i] = knowledge[character][i];
                        }
                    }
                    4 => { // Sakuya random daggers and time stop
                        /*
                        if arg < 2 {
                            drop_particle(&PARTICLES_ARRAY,0xc,enemy->pos,1,0xffffffff);
                            //TODO: is that the timestop?
                            LEVEL.field_0x2c = arg;
                            return;
                        }
                        // this changes the orientation of random bullets
                        let mut max_bullets = 0xe;
                        if (LEVEL.rank >= 2) {max_bullets = 0x34;}
                        i = 0;
                        for bullet in game.bullets {
                            if bullet->state != 0 && bullet->state != 5 && 30. <= (bullet->sprites[0].additional_infos)->height && bullet->field_0x5ba != 5 && (uVar3 = prng16(&PRNG_STATE), (uVar3 & 3) == 0) {
                                bullet->field_0x5ba = 5;
                                new_effect(GAME_OBJECT,(sprite *)bullet, (int)bullet->sprites[0].sometimes_copy_of_UNK1 + (int)bullet->field_0x5ba);
                                x = bullet->pos[0] - PLAYER.pos[0];
                                y = bullet->pos[1] - PLAYER.pos[1];
                                if sqrt(x*x+y*y) > 128. {
                                    if LEVEL.rank >= 2 {bullet->automatic_angle = prng_double() * 2*pi;}
                                    else{bullet->automatic_angle = (prng_double() * ((pi/8)*6) + pi/4);}
                                    else {
                                        // TODO: check player_get_angle, might be what ST0 is
                                        player_get_angle_to(&PLAYER,bullet->pos,local_38);
                                        bullet->automatic_angle = (extraout_ST0_00 + pi/2 + (prng_double() * pi*2));
                                    }
                                    bullet->dpos[0] = cos(bullet->automatic_angle) * bullet->speed;
                                    bullet->dpos[1] = sin(bullet->automatic_angle) * bullet->speed;
                                    max_bullets -= 1;
                                    if (max_bullets == 0) break;
                                }
                            }
                            (enemy->ecl_frame).var_ints_1[2] = 0;*/
                        }
                    7 => { // Remilia's lazer maze
                        // so what this does is kinda complex: 2 rounds of 3 subrounds of 8 shots, either
                        // laser or standard bullets depending on the argument passed.
                        // it is done in 2 steps: first we precalculate coordinates of the 8 shots for the first subround
                        // set the shot properties depending on difficulties and current round and then
                        // edit  the coordinates for the next round
                        let rnd_pos = self.get_prng().borrow_mut().get_f64() * 2. * std::f64::consts::PI;
                        let enemy = self.enemy.borrow();
                        for i in 0..2 {
                            let mut pos: [f64; 8*3] = [0.; 8*3];
                            let mut offset = rnd_pos -((std::f64::consts::PI/8.)*7.);
                            let mut next_offset = -std::f64::consts::PI/4.;
                            if (i == 0) {
                                offset = rnd_pos -std::f64::consts::PI;
                                next_offset = std::f64::consts::PI/4.;
                            }

                            // we calculate angle, speed and offset for the 8 shots
                            let mut offset_copy=offset;
                            for y in 0..8 {
                                pos[y * 3] = offset_copy.cos() * 32. + enemy.pos.x as f64;
                                pos[y * 3 + 1] = offset_copy.sin() * 32. + enemy.pos.y as f64;
                                pos[y * 3 + 2] = enemy.z as f64;
                                offset_copy += std::f64::consts::PI/4.;
                            }

                            // 3 rounds of 8 shots
                            for z in 0..3 {

                                let mut length = 112.;
                                // last subround
                                if (z == 2) {length = 480.;}

                                for y in 0..8 {
                                    /*
                                    if (arg == 0) {
                                        let (mut si, mut ged, mut ed) = (8, 20.,ed=430.);
                                        if (LEVEL.rank < 2) {si=2; ged=28.; ed=length;}
                                        laser_args.angle = pos[y * 3];
                                        laser_args.speed = pos[y * 3 + 1];
                                        laser_args.start_offset = pos[y * 3 + 2];
                                        laser_args.type = 1;
                                        laser_args.sprite_index_offset = si;
                                        laser_args.end_offset = offset;
                                        laser_args.width = 0.;
                                        laser_args.duration = 0;
                                        laser_args.grazing_extra_duration = ged;
                                        laser_args.end_duration = ed;
                                        laser_args.UNK1 = z * 0x10 + 0x3c;
                                        laser_args.grazing_delay = laser_args.end_duration;
                                        fire_laser(&ETAMA_ARRAY,&laser_args);
                                    }
                                    else {
                                        (enemy->bullet_attributes).pos[0] = pos[y * 3];
                                        (enemy->bullet_attributes).pos[1] = pos[y*3+1];
                                        (enemy->bullet_attributes).pos[2] = pos[y*3+2];
                                        bullet_fire(&enemy->bullet_attributes,&ETAMA_ARRAY);
                                    }
                                    */
                                    pos[y * 3] = offset.cos() * length + pos[y * 3];
                                    pos[y * 3 + 1] = offset.sin() * length + pos[y * 3 + 1];
                                    offset = offset + std::f64::consts::PI/4.;
                                }
                                offset = (next_offset - 2.*std::f64::consts::PI) + offset;
                            }
                        }
                    }
                    8 => { // Vampire Fantasy
                        let n = {
                            let enemy = self.enemy.borrow();
                            let game = enemy.game.upgrade().unwrap();
                            let mut game = game.borrow_mut();
                            let mut n = 0;
                            for bullet in game.bullets.iter() {
                                let mut bullet = bullet.borrow();
                                // TODO: uncomment that one.
                                if bullet.state != 0 && bullet.state != 5 /* && (30. <= (bullet.sprites[0].additional_infos).height) */ {
                                    let prng = enemy.prng.upgrade().unwrap();
                                    let random = prng.borrow_mut().get_f64();
                                    let launch_angle = (random * (2. * std::f64::consts::PI) - std::f64::consts::PI) as f32;
                                    let mut attribs = BulletAttributes {
                                        // TODO: check if the z value of this pos is really used.
                                        pos: bullet.pos,
                                        anim: 3,
                                        sprite_index_offset: 1,
                                        launch_angle,
                                        speed: 0.,
                                        angle: 0.,
                                        speed2: 0.,
                                        bullets_per_shot: 1,
                                        number_of_shots: 1,
                                        flags: 8,
                                        bullet_type: 1,
                                        extended_attributes: Default::default(),
                                        sound: None,
                                    };
                                    attribs.fire();
                                    n += 1
                                }
                            }
                            n
                        };
                        //TODO: this variable might not always be correct! it uses the argument in
                        //th06: *(int *)(param_1 + 0x9b0) = local_60;
                        self.set_i32(-10004, n);
                    }

                    9 => {
                        let mut rnd = self.get_prng().borrow_mut().get_f64();
                        //TODO: the game does that
                        //drop_particle(&PARTICLES_ARRAY,0xc,enemy->pos,1,0xffffffff);
                        //self._game.new_effect((enemy.x, enemy.y), 17)
                        /*
                        for bullet in game.bullets {
                            if bullet._bullet_type.state != 0 && bullet._bullet_type.state != 5 && (30. <= (bullet.sprites[0].additional_infos)->height) && bullet.speed == 0. {
                                bullet.flags = bullet.flags | 0x10;
                                //TODO: reverse this field and effect
                                bullet->field_0x5ba = 2;
                                new_effect(GAME_OBJECT,(sprite *)bullet, (int)bullet->sprites[0].sometimes_copy_of_UNK1 + (int)bullet->field_0x5ba);
                                bullet.speed=0.01;
                                bullet.frame=0x78;

                                let mut dx = bullet.x - enemy.x;
                                let mut distance = dx.hypot(bullet.y - enemy.y);

                                if distance > 0.01 {
                                    distance = sqrt(distance);
                                }else{distance = 0.;}
                                let mut angle = (distance * std::f64::consts::PI) / 256. + (rnd * (2*std::f64::consts::PI) - std::f64::consts::PI);
                                bullet->attributes[0] = cos(angle) * 0.01000000;
                                bullet->attributes[1] = sin(angle) * 0.01000000;
                            }
                        }
                        */
                    }
                    11 => {
                        self.get_prng().borrow_mut().get_f64();
                        //TODO: the game does that
                        //drop_particle(&PARTICLES_ARRAY,0xc,enemy->pos,1,0xffffffff);
                        //self._game.new_effect((enemy.x, enemy.y), 17)
                        /*
                        for bullet in game.bullets {
                            if bullet._bullet_type.state != 0 && bullet._bullet_type.state != 5 && (30. <= (bullet.sprites[0].additional_infos)->height) && bullet.speed == 0. {
                                bullet.flags = bullet.flags | 0x10;
                                //TODO: reverse this field and effect
                                bullet->field_0x5ba = 2;
                                new_effect(GAME_OBJECT,(sprite *)bullet, (int)bullet->sprites[0].sometimes_copy_of_UNK1 + (int)bullet->field_0x5ba);
                                bullet.speed=0.01;
                                bullet.frame=0x78;
                                let mut angle = self.get_prng().borrow_mut().get_f64() * (2*std::f64::consts::PI) - std::f64::consts::PI;
                                bullet->attributes[0] = cos(angle) * 0.01000000;
                                bullet->attributes[1] = sin(angle) * 0.01000000;


                            }
                        }
                        */
                    }
                    13 => {
                        if self.frame.ints1[3] % 6 == 0 {
                            let mut _angle=self.frame.floats[2];
                            /*
                            (type_, anim, sprite_idx_offset, bullets_per_shot, number_of_shots,
                            speed, speed2, launch_angle, angle, flags) = self._enemy.bullet_attributes
                            for i in range(arg) {
                                //TODO: distance is obtained directly by copying bullet attributes
                                //in memory
                                launch_pos = (192 + cos(_angle) * _distance,
                                    224 + sin(_angle) * _distance);

                                bullet_attributes = (type_, anim, sprite_idx_offset,
                                     bullets_per_shot, number_of_shots,
                                     speed, speed2,
                                     _angle + self.frame.floats[1], angle, flags);
                                enemy.fire(launch_pos=launch_pos,bullet_attributes=bullet_attributes);
                                _angle += 2*std::f64::consts::PI/arg;
                            }*/
                        }
                        self.frame.ints1[3] += 1;
                    }
                    14 => { // Lävatein
                        let mut enemy = self.enemy.borrow_mut();
                        self.frame.ints1[3] = 0;
                        for laser in enemy.laser_by_id.values() {
                            //for pos in laser.get_bullets_pos(){
                            //TODO: the game checks for laser end_offset before firing
                            //  enemy.fire(launch_pos=pos);
                            //}
                            self.frame.ints1[3] += 1;
                        }
                    }
                    16 => { // QED: Ripples of 495 years
                        let mut enemy = self.enemy.borrow_mut();
                        let game = enemy.game.upgrade().unwrap();
                        let mut game = game.borrow_mut();
                        if arg == 0 {
                            self.frame.floats[3] = 2. - (enemy.life as f32) / 6000.;
                            self.frame.ints2[1] = ((enemy.life * 240) / 6000 + 40) as i32;
                        } else {
                            let fx = (320. - ((enemy.life as f32) * 160.) / 6000.) as f64;
                            let fy = (128. - ((enemy.life as f32) * 64.) / 6000.) as f64;
                            let rand_x = game.prng.borrow_mut().get_f64();
                            let rand_y = game.prng.borrow_mut().get_f64();
                            self.frame.floats[2] = (rand_x * fx + (192. - fx / 2.)) as f32;
                            self.frame.floats[3] = (rand_y * fy + (96. - fy / 2.)) as f32;
                        }
                    }
                    _ => unimplemented!("Special function {:?} not found!", function)
                }
            }

            // 122
            // Here lies the Di Sword of despair
            SubInstruction::SetSpecialFunctionCallback(function) => {
                //TODO: those functions are executed at each frame and needs to be written to the
                //callback function but so far i'm simply focusing on the implementation
                //NB: the original engine doesn't differenciate between function blocks for ins 121
                //and 122 but we do here, since it wouldn't make sense the other way around.
                match function {
                    12 => {
                        for i in 0..8 {
                            /*
                            if ((enemy->lasers[i] != (laser *)0x0) && (enemy->lasers[i]->state != 0)) {
                                (enemy->bullet_attributes).pos[0] = cos(enemy->lasers[i]->angle) * 64. + enemy.pos.x;
                                // yes, it reads pos[0] after it has been modified and yes, this most
                                // likely is a bug
                                (enemy->bullet_attributes).pos[1] = cos(enemy->lasers[i]->angle) * (enemy->bullet_attributes).pos[0] + enemy.pos.y;
                                (enemy->bullet_attributes).pos[2] = enemy.z;
                                bullet_fire(&enemy->bullet_attributes,&ETAMA_ARRAY);
                            }*/
                        }
                    }
                    _ => unimplemented!("Special function {:?} not found!", function)
                }
            }
            _ => unimplemented!("{:?}", instruction)
        }

    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::th06::anm0::Anm0;
    use crate::th06::ecl::{Sub, CallSub, Rank};
    use crate::th06::enemy::Game;
    use std::io::{self, Read};
    use std::fs::File;

    fn setup() -> (Rc<RefCell<Game>>, Rc<RefCell<Enemy>>) {
        let file = File::open("EoSD/ST/stg1enm.anm").unwrap();
        let mut file = io::BufReader::new(file);
        let mut buf = vec![];
        file.read_to_end(&mut buf).unwrap();
        let (_, mut anms) = Anm0::from_slice(&buf).unwrap();
        let anm0 = anms.pop().unwrap();
        let anm0 = Rc::new(RefCell::new(anm0));
        let prng = Rc::new(RefCell::new(Prng::new(0)));
        let game = Game::new(prng, Rank::EASY);
        let game = Rc::new(RefCell::new(game));
        let enemy = Enemy::new(Position::new(0., 0.), 500, 0, 640, Rc::downgrade(&anm0), Rc::downgrade(&game));
        (game, enemy)
    }

    #[test]
    fn call_and_return() {
        let (game, enemy) = setup();
        let ecl = Ecl { mains: vec![], subs: vec![
            Sub { instructions: vec![
                CallSub::new(0, Rank::EASY, SubInstruction::Call(1, 13, 12.)),
            ]},
            Sub { instructions: vec![
                CallSub::new(0, Rank::EASY, SubInstruction::Noop()),
                CallSub::new(1, Rank::EASY, SubInstruction::Return()),
            ]},
        ]};
        let mut ecl_runner = EclRunner::new(&ecl, enemy, 0);
        ecl_runner.run_frame();
        assert_eq!(ecl_runner.frame.ints1[0], 13);
        assert_eq!(ecl_runner.frame.floats[0], 12.);
        assert_eq!(ecl_runner.stack.len(), 1);
        ecl_runner.run_frame();
        assert_eq!(ecl_runner.frame.ints1[0], 0);
        assert_eq!(ecl_runner.frame.floats[0], 0.);
        assert_eq!(ecl_runner.stack.len(), 0);
    }
}
