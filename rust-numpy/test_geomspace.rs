fn main() {
    let start = 1.0_f32;
    let stop = 1000.0_f32;
    let num = 4_usize;
    let endpoint = true;

    let start_log = start.ln();
    let stop_log = stop.ln();

    let div = if endpoint {
        (num - 1) as f32
    } else {
        num as f32
    };

    let step = (stop_log - start_log) / div;

    println!("start_log: {}", start_log);
    println!("stop_log: {}", stop_log);
    println!("step: {}", step);

    for i in 0..num {
        let lin = start_log + (i as f32) * step;
        let geo = lin.exp();
        println!("i={}, lin={}, geo={}", i, lin, geo);
    }
}
