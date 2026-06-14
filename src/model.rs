use anyhow::{Result, anyhow};
use num_rational::Ratio;
use std::fmt::{Display, Write};

use symbolica::evaluate::{Instruction, Slot};
use symbolica::prelude::*;

fn slot(slot: &Slot) -> String {
    match slot {
        Slot::Param(idx) => format!("('param', {})", idx),
        Slot::Temp(idx) => format!("('temp', {})", idx),
        Slot::Out(idx) => format!("('out', {})", idx),
        Slot::Const(idx) => format!("('const', {})", idx),
    }
}

fn slots(slots: &[Slot]) -> String {
    let mut buf = format!("[{}", slot(&slots[0]));

    for s in slots.iter().skip(1) {
        write!(buf, ", {}", slot(s)).unwrap();
    }

    write!(buf, "]").unwrap();
    buf
}

fn boolean(b: bool) -> String {
    if b {
        "True".to_string()
    } else {
        "False".to_string()
    }
}

pub fn write_instructions(prog: ExportedInstructions<Complex<f64>>) -> Result<String> {
    let mut buf = "([".to_string();

    for q in prog.instructions.iter() {
        match q {
            Instruction::Add(lhs, args, num_reals) => {
                writeln!(
                    buf,
                    "('add', {}, {}, {}),",
                    slot(&lhs),
                    slots(&args),
                    num_reals
                )
                .unwrap();
            }
            Instruction::Mul(lhs, args, num_reals) => {
                writeln!(
                    buf,
                    "('mul', {}, {}, {}),",
                    slot(&lhs),
                    slots(&args),
                    num_reals
                )
                .unwrap();
            }
            Instruction::Pow(lhs, arg, p, is_real) => {
                writeln!(
                    buf,
                    "('pow', {}, {}, {}, {}),",
                    slot(&lhs),
                    slot(&arg),
                    p,
                    boolean(*is_real)
                )
                .unwrap();
            }
            Instruction::Powf(lhs, arg, p, is_real) => {
                writeln!(
                    buf,
                    "('powf', {}, {}, {}, {}),",
                    slot(&lhs),
                    slot(&arg),
                    slot(&p),
                    boolean(*is_real)
                )
                .unwrap();
            }
            Instruction::Assign(lhs, rhs) => {
                writeln!(buf, "('assign', {}, {}),", slot(&lhs), slot(&rhs)).unwrap();
            }
            Instruction::Fun(lhs, fun, is_real) => {
                writeln!(
                    buf,
                    "('fun', {}, '{}', [], {}, {}),",
                    slot(&lhs),
                    fun.0
                        .get_ascii_name()
                        .unwrap()
                        .strip_prefix("symbolica_")
                        .unwrap(),
                    slots(&fun.2),
                    boolean(*is_real)
                )
                .unwrap();
            }
            Instruction::Join(lhs, cond, true_val, false_val) => {
                writeln!(
                    buf,
                    "('join', {}, {}, {}, {}),",
                    slot(&lhs),
                    slot(&cond),
                    slot(&true_val),
                    slot(&false_val)
                )
                .unwrap();
            }
            Instruction::Label(id) => {
                writeln!(buf, "('label', {}),", id).unwrap();
            }
            Instruction::IfElse(cond, id) => {
                writeln!(buf, "('if_else', {}, {}),", slot(&cond), id).unwrap();
            }
            Instruction::Goto(id) => {
                writeln!(buf, "('goto', {}),", id).unwrap();
            }
        }
    }

    let c: Vec<String> = prog
        .constants
        .iter()
        .map(|x| rationalize_complex(*x).unwrap())
        .collect();

    write!(
        buf,
        "(label, 0)], {}, [{}])",
        &prog.temporary_count,
        c.join(", ")
    )
    .unwrap();
    Ok(buf)
}

fn rationalize_complex(x: Complex<f64>) -> Result<String> {
    let (x, imaginary) = if x.im == 0.0 {
        (x.re, false)
    } else if x.re == 0.0 {
        (x.im, true)
    } else {
        return Err(anyhow!(
            "only pure real or imaginary constants are allowed."
        ));
    };

    let r: Ratio<i64> = Ratio::approximate_float(x).unwrap();

    if imaginary {
        if *r.denom() == 1 {
            Ok(format!("{}𝑖", r.numer()))
        } else {
            Ok(format!("{}𝑖/{}", r.numer(), r.denom()))
        }
    } else {
        if *r.denom() == 1 {
            Ok(format!("{}", r.numer()))
        } else {
            Ok(format!("{}/{}", r.numer(), r.denom()))
        }
    }
}
