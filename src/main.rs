use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Indeterminate, Symbol},
    domains::{
        float::Complex,
        integer::IntegerRing,
        rational::{Fraction, Rational},
    },
    evaluate::{
        BatchEvaluator, CompileOptions, ExportSettings, ExpressionEvaluator, FunctionMap,
        InlineASM, JITCompilationSettings, OptimizationSettings,
    },
    id::{MatchSettings, Replacement},
    parse_lit, symbol, try_parse,
};

use std::time::Instant;

mod model;
use crate::model::*;

const N: usize = 10000;

const DEFAULT_PAYLOAD: &str = "payload.json";
const DEFAULT_ARTIFACT_DIR: &str = "artifacts";
const MISMATCH_TOLERANCE: f64 = 1.0e-24;

#[derive(Deserialize)]
struct Payload {
    description: String,
    graph_name: String,
    stack_label: String,
    method: String,
    function_name: String,
    param_builder_params: Vec<String>,
    fn_map_entries: Vec<(String, String, Vec<String>, Vec<String>)>,
    exprs: Vec<String>,
    additional_fn_map_entries: Vec<(String, String, Vec<String>, Vec<String>)>,
    input: Vec<[f64; 2]>,
}

type ParsedFnMapEntry = (Atom, Atom, Vec<Atom>, Vec<Indeterminate>);

fn parse_fn_map_entries(
    entries: &[(String, String, Vec<String>, Vec<String>)],
) -> Result<Vec<ParsedFnMapEntry>> {
    entries
        .iter()
        .map(|(lhs, rhs, tags, args)| {
            let lhs_atom = try_parse!(lhs).map_err(|e| anyhow!(e))?;
            let rhs_atom = try_parse!(rhs).map_err(|e| anyhow!(e))?;
            let tags = tags
                .iter()
                .map(|tag| try_parse!(tag).map_err(|e| anyhow!(e)))
                .collect::<Result<Vec<_>>>()?;
            let args = args
                .iter()
                .map(|arg| {
                    let arg = try_parse!(arg).map_err(|e| anyhow!(e))?;
                    arg.clone()
                        .try_into()
                        .map_err(|_| anyhow!("expected indeterminate, got {arg}"))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok((lhs_atom, rhs_atom, tags, args))
        })
        .collect()
}

fn apply_fn_map_entries(
    parsed_entries: Vec<ParsedFnMapEntry>,
) -> Result<(Vec<Replacement>, FunctionMap)> {
    let mut fn_map = FunctionMap::new();
    let mut replacements = Vec::new();
    let v: Vec<Symbol> = Vec::new();
    fn_map.add_function(symbol!("x"), v, Atom::Zero)?;

    for (lhs, rhs, tags, args) in parsed_entries {
        if let AtomView::Var(_) = lhs.as_view() {
            if let Ok(_) = Complex::<Rational>::try_from(rhs.as_view()) {
                let v: Vec<Symbol> = Vec::new();
                fn_map.add_function(lhs.get_symbol().unwrap(), v, rhs)?;
            } else {
                replacements.push(Replacement::new(lhs.to_pattern(), rhs.clone()));
            }
        } else if let AtomView::Fun(f) = lhs.as_view() {
            if tags.is_empty() {
                let wildcards = args
                    .iter()
                    .enumerate()
                    .map(|(i, arg)| {
                        let atom: Atom = arg.clone().into();
                        Replacement::new(
                            atom.to_pattern(),
                            Atom::var(symbolica::symbol!(format!("x{i}_"))),
                        )
                        .allow_new_wildcards_on_rhs(true)
                    })
                    .collect::<Vec<_>>();

                fn_map
                    .add_function(f.get_symbol(), args.clone(), rhs.clone())
                    .map_err(|e| anyhow!(e))?;

                replacements.push(Replacement::new(
                    lhs.replace_multiple(&wildcards).to_pattern(),
                    rhs.replace_multiple(&wildcards),
                ));
            } else {
                fn_map
                    .add_tagged_function(f.get_symbol(), tags.clone(), args.clone(), rhs.clone())
                    .map_err(|e| anyhow!(e))?;
            }
        } else {
            replacements.push(Replacement::new(lhs.to_pattern(), rhs.clone()));
        }
    }

    Ok((replacements, fn_map))
}

fn build_evaluator(payload: &Payload) -> Result<ExpressionEvaluator<Complex<f64>>> {
    let params = payload
        .param_builder_params
        .iter()
        .map(|param| try_parse!(param).map_err(|e| anyhow!(e)))
        .collect::<Result<Vec<_>>>()?;
    let exprs = payload
        .exprs
        .iter()
        .map(|expr| try_parse!(expr).map_err(|e| anyhow!(e)))
        .collect::<Result<Vec<_>>>()?;

    let mut fn_map_entries = parse_fn_map_entries(&payload.fn_map_entries)?;
    fn_map_entries.extend(parse_fn_map_entries(&payload.additional_fn_map_entries)?);
    let (replacements, fn_map) = apply_fn_map_entries(fn_map_entries)?;

    Atom::evaluator_multiple(
        &exprs
            .iter()
            .map(|expr| expr.replace_multiple(&replacements))
            .collect::<Vec<_>>(),
        &params,
    )
    .function_map(fn_map)
    .build()
    .map(|eval| {
        eval.map_coeff(&|r: &Complex<Fraction<IntegerRing>>| {
            Complex::new(r.re.to_f64(), r.im.to_f64())
        })
    })
    .map_err(|e| anyhow!(e))
}

fn input_values(payload: &Payload) -> Vec<Complex<f64>> {
    payload
        .input
        .iter()
        .map(|[re, im]| Complex::new(*re, *im))
        .collect()
}

fn eval_eager(
    eval: &mut ExpressionEvaluator<Complex<f64>>,
    input: &[Complex<f64>],
) -> Result<(Complex<f64>, f64)> {
    let mut out = vec![Complex::new(0.0, 0.0); 1];

    let t0 = Instant::now();

    for _ in 0..N {
        eval.evaluate(input, &mut out);
    }

    let duration = t0.elapsed().as_secs_f64();

    Ok((out[0], duration))
}

fn eval_symjit(
    eval: &mut ExpressionEvaluator<Complex<f64>>,
    input: &[Complex<f64>],
    direct: bool,
) -> Result<(Complex<f64>, f64)> {
    let eval = eval.clone().map_coeff(&|z| Complex::new(z.re, z.im));

    let code = eval.export_instructions();
    let l = write_instructions(code).unwrap();
    std::fs::write("mre_gl15.txt", l)?;

    let config = JITCompilationSettings::default().direct_translation(direct);
    let mut app = eval.jit_compile(config).unwrap();

    let input: Vec<Complex<f64>> = input.iter().map(|z| Complex::new(z.re, z.im)).collect();
    println!("{:?}", &input);
    let mut out = vec![Complex::new(0.0, 0.0); 1];

    let t0 = Instant::now();

    for _ in 0..N {
        app.evaluate(&input, &mut out);
    }

    let duration = t0.elapsed().as_secs_f64();

    Ok((out[0], duration))
}

fn eval_symjit_batch(
    eval: &mut ExpressionEvaluator<Complex<f64>>,
    input: &[Complex<f64>],
    direct: bool,
) -> Result<(Complex<f64>, f64)> {
    let eval = eval.clone().map_coeff(&|z| Complex::new(z.re, z.im));

    let config = JITCompilationSettings::default().direct_translation(direct);
    let mut app = eval.jit_compile(config).unwrap();

    let input: Vec<Complex<f64>> = input.iter().map(|z| Complex::new(z.re, z.im)).collect();
    let l = input.len();

    let mut args: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); N * l];

    for i in 0..N {
        args[i * l..(i + 1) * l].copy_from_slice(&input);
    }

    let mut out = vec![Complex::new(0.0, 0.0); N];

    let t0 = Instant::now();

    let _ = app.evaluate_batch(N, &args, &mut out);

    let duration = t0.elapsed().as_secs_f64();

    Ok((out[0], duration))
}

fn eval_assembly(
    eval: &mut ExpressionEvaluator<Complex<f64>>,
    input: &[Complex<f64>],
    artifact_dir: &Path,
    function_name: &str,
) -> Result<(Complex<f64>, f64)> {
    fs::create_dir_all(artifact_dir)?;
    let cpp = artifact_dir.join(format!("{function_name}.cpp"));
    let so = artifact_dir.join(format!("{function_name}.so"));
    let mut compiled = eval
        .export_cpp::<Complex<f64>>(
            &cpp,
            function_name,
            ExportSettings::new()
                .include_header(true)
                .inline_asm(InlineASM::X64)
                .custom_header(None),
        )
        .map_err(|e| anyhow!(e))?
        .compile(&so, CompileOptions::default())
        .map_err(|e| anyhow!(e))?
        .load()
        .map_err(|e| anyhow!(e))?;
    let mut out = vec![Complex::new(0.0, 0.0); 1];

    let t0 = Instant::now();

    for _ in 0..N {
        compiled.evaluate(input, &mut out);
    }

    let duration = t0.elapsed().as_secs_f64();

    Ok((out[0], duration))
}

fn max_abs_diff(lhs: Complex<f64>, rhs: Complex<f64>) -> f64 {
    (lhs.re - rhs.re).abs().max((lhs.im - rhs.im).abs())
}

fn main() -> Result<()> {
    let payload_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PAYLOAD));
    let payload: Payload = serde_json::from_slice(&fs::read(&payload_path)?)?;
    let artifact_dir = payload_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(DEFAULT_ARTIFACT_DIR);
    let input = input_values(&payload);

    println!("description: {}", payload.description);
    println!(
        "graph={} stack={} method={} expr_len={} params={} input_len={}",
        payload.graph_name,
        payload.stack_label,
        payload.method,
        payload.exprs[0].len(),
        payload.param_builder_params.len(),
        input.len()
    );
    println!("expression: {}", payload.exprs[0]);

    let mut eager_eval = build_evaluator(&payload)?;
    let (eager, t1) = eval_eager(&mut eager_eval, &input)?;
    println!(
        "eager   = {eager}\n in {:.2} μsec",
        1000000.0 * t1 / (N as f64)
    );

    /*
    let mut assembly_eval = build_evaluator(&payload)?;
    let (assembly, t2) = eval_assembly(
        &mut assembly_eval,
        &input,
        &artifact_dir,
        &payload.function_name,
    )?;

    println!(
        "assembly= {assembly}\n in {:.2} μsec",
        1000000.0 * t2 / (N as f64)
    );
    */

    let mut symjit_eval = build_evaluator(&payload)?;

    let (symjit_direct, td0) = eval_symjit(&mut symjit_eval, &input, true)?;
    let (symjit_indirect, ti0) = eval_symjit(&mut symjit_eval, &input, false)?;

    let (symjit_direct_batch, td1) = eval_symjit_batch(&mut symjit_eval, &input, true)?;
    let (symjit_indirect_batch, ti1) = eval_symjit_batch(&mut symjit_eval, &input, false)?;

    println!(
        "symjit  = {symjit_direct}\n in {:.2} μsec (direct)",
        1000000.0 * td0 / (N as f64)
    );

    println!(
        "symjit  = {symjit_indirect}\n in {:.2} μsec (indirect)",
        1000000.0 * ti0 / (N as f64)
    );

    println!(
        "symjit  = {symjit_direct_batch}\n in {:.2} μsec (direct; batch)",
        1000000.0 * td1 / (N as f64)
    );

    println!(
        "symjit  = {symjit_indirect_batch}\n in {:.2} μsec (indirect; batch)",
        1000000.0 * ti1 / (N as f64)
    );

    Ok(())
}
