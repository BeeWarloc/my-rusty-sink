extern crate num_complex;
extern crate rustfft;
extern crate num_traits;

use num_traits::identities::Zero;
use num_complex::Complex64 as c64;
use std::f64::consts::PI;
use std::usize;
use std::u32;



fn convolve_in_time_domain(f: &[f64], g: &[f64]) -> Vec<f64> {
    let mut out = vec![0f64; f.len() + g.len() - 1];
    for f_idx in 0..(f.len()) {
        for g_idx in 0..(g.len()) {
            out[f_idx + g_idx] += f[f_idx] * g[g_idx];
        }
    }
    out
}

// NOT YET WORKING BOBO
// SCALING ISSUES!!!!!!
// https://dsp.stackexchange.com/questions/43337/how-to-scale-the-fft-when-using-it-to-calculate-discrete-convolution
fn convolve_in_frequency_domain(f: &[f64], g: &[f64]) -> Vec<f64> {
    // first find an ok fft size
    let fft_len = g.len().next_power_of_two() * 2;
    let chunk_len = (fft_len - g.len()) + 1;
    let mut out = vec![0.0f64; f.len() + g.len() - 1];

    // get dft of g
    let mut g_padded = Vec::from(g);
    g_padded.resize(fft_len, 0.0);
    // _slightly_ stupid:
    let g_scale_factor = (fft_len as f64).sqrt();
    let g_padded: Vec<_> = g_padded.into_iter().map(|x| c64::new(x * g_scale_factor, 0.0)).collect();
    let g_freq_domain = complex_dft(&g_padded);
    // chunk up, then overlap add

    let mut out_offset = 0;
    for chunk in f.chunks(chunk_len) {
        let mut chunk_padded = Vec::from(chunk);
        chunk_padded.resize(fft_len, 0.0);
        let chunk_padded: Vec<_> = chunk_padded.into_iter().map(|x| c64::new(x, 0.0)).collect();
        let chunk_freq_domain = complex_dft(&chunk_padded);

        // Convolution in frequency domain by multiplication
        assert!(chunk_freq_domain.len() == g_freq_domain.len());
        let res_freq_domain: Vec<_> = g_freq_domain.iter().zip(chunk_freq_domain.iter()).map(|(a,b)| *a * *b ).collect();
        let res = complex_idft(&res_freq_domain);

        for i in 0..(chunk.len() + g.len() - 1) {
            out[out_offset + i] += res[i].re;
        }

        out_offset += chunk_len;
    }

    out
}

// 0 1 2 3 4 5 6 7
// 0 | 1 | 2 | 3 |  x*2+0
// 0 | | | 1 | | |  x*4+0
// | | 0 | | | 1 |  x*4+2
// 0 | | | | | | |  x*8+0

// | 0 | 1 | 2 | 3  x*2+1
// | 0 | | | 1 | |  x*4+1
// | | | 0 | | | 1  x*4+3
// | | | 0 | | | |  x*8+3
// | | | | | | | 0  x*8+7

fn reverse_bits_u32(x: u32) -> u32 {
    let x = ((x & 0xaaaaaaaa) >> 1) | ((x & 0x55555555) << 1);
    let x = ((x & 0xcccccccc) >> 2) | ((x & 0x33333333) << 2);
    let x = ((x & 0xf0f0f0f0) >> 4) | ((x & 0x0f0f0f0f) << 4);
    let x = ((x & 0xff00ff00) >> 8) | ((x & 0x00ff00ff) << 8);
    ((x >> 16) | (x << 16))
}

fn reverse_bits(x: u32, bit_count: u32) -> u32 {
    reverse_bits_u32(x) >> (32 - bit_count)
}

pub fn fill_reverse_bits_lut(array: &mut [u32]) {
    assert!(array.len().is_power_of_two() || array.len() > u32::MAX as usize);
    let bit_count = array.len().trailing_zeros();
    for i in 0..array.len() {
        array[i] = reverse_bits(i as u32, bit_count);
    }
}

pub fn complex_ifft(input: &[c64]) -> Vec<c64> {
    if input.len() == 1 {
        vec![input[0]]
    } else {
        let even: Vec<_> = input.iter().enumerate().filter_map(|(i,x)| if i & 1 == 0 { Some(*x) } else { None }).collect();
        let odd: Vec<_> = input.iter().enumerate().filter_map(|(i,x)| if i & 1 == 1 { Some(*x) } else { None }).collect();
        let even = complex_ifft(&even);
        let odd = complex_ifft(&odd);
        ifft_combine(&even, &odd)
    }
}

pub fn ifft_combine(even: &[c64], odd: &[c64]) -> Vec<c64> {
    assert!(even.len() == odd.len());
    let i = c64::i();
    let n = (even.len() * 2) as f64;
    let mut out = vec![c64::zero(); even.len() * 2];
    // orthonormal scaling, makes scaling the same for the forward and inverse transform
    let scale_factor = 1.0 / 2.0f64.sqrt();
    for k in 0..even.len() {
        let e = even[k];
        let o = odd[k];
        // Twiddle factor, ie rotate by one sample(?)
        let w = (i * 2.0 * PI * (k as f64) / n).exp();
        out[k           ] = (e + w * o) * scale_factor;
        out[k+even.len()] = (e - w * o) * scale_factor;
    }
    out
}

// This is _absolutely_ not for performance, just for verifying my own understanding of the concept
pub fn complex_fft(input: &[c64]) -> Vec<c64> {
    if input.len() == 1 {
        vec![input[0]]
    } else {
        let even: Vec<_> = input.iter().enumerate().filter_map(|(i,x)| if i & 1 == 0 { Some(*x) } else { None }).collect();
        let odd: Vec<_> = input.iter().enumerate().filter_map(|(i,x)| if i & 1 == 1 { Some(*x) } else { None }).collect();
        let even = complex_fft(&even);
        let odd = complex_fft(&odd);
        fft_combine(&even, &odd)
    }
}

pub fn fft_combine(even: &[c64], odd: &[c64]) -> Vec<c64> {
    assert!(even.len() == odd.len());
    let i = c64::i();
    let n = (even.len() * 2) as f64;
    let mut out = vec![c64::zero(); even.len() * 2];
    let scale_factor = 1.0 / 2.0f64.sqrt();
    for k in 0..even.len() {
        let e = even[k];
        let o = odd[k];
        // Twiddle factor, ie rotate by one sample(?)
        let w = (-i * 2.0 * PI * (k as f64) / n).exp();
        out[k           ] = (e + w * o) * scale_factor;
        out[k+even.len()] = (e - w * o) * scale_factor;
    }
    out
}

// ifft: x[j] = sum(y[k] exp(-2 %i %pi j k / n), k, 0, n - 1)
pub fn complex_idft(y: &[c64]) -> Vec<c64> {
    let n = y.len();
    let i = c64::i();
    let mut x = Vec::with_capacity(n);
    let scale_factor = 1.0 / (n as f64).sqrt();
    for k in 0..n {
        x.push(y.iter().enumerate().map(|(j, xj)| {
            xj * (i * 2.0 * PI * (k as f64) * (j as f64) / (n as f64)).exp()
        }).fold(c64::new(0.0,0.0), |acc, v| acc + v) * scale_factor);
    }
    x
}

// fft : y[k] = (1/n) sum(x[j] exp(+2 %i %pi j k / n), j, 0, n - 1)
pub fn complex_dft(x: &[c64]) -> Vec<c64> {
    let n = x.len();
    let i = c64::i();
    let mut y = Vec::with_capacity(n);
    let scale_factor = 1.0 / (n as f64).sqrt();
    for k in 0..n {
        y.push(x.iter().enumerate().map(|(j, xj)| {
            xj * (-i * 2.0 * PI * (k as f64) * (j as f64) / (n as f64)).exp()
        }).fold(c64::new(0.0,0.0), |acc, v| acc + v) * scale_factor);
    }
    y
}


pub fn real_idft(fd: &[c64]) -> Vec<f64> {
    let mut x = Vec::new();
    let fd_end = fd.len() - 1;
    let n = fd_end * 2;
    let fd_scale_factor = n as f64 / 2.0;
    let mut fd: Vec<_> = fd.iter().map(|c| {
        *c
    }).collect();

    // Special cases
    fd[0].re /= 2.0;
    fd[fd_end].re /= 2.0;

    let inv_n = 1.0 / n as f64;

    for i in 0..n {
        let mut sum = 0.0;
        for (k, c) in fd.iter().enumerate() {
            sum +=
                c.re * ((2.0 * PI * (k as f64) * (i as f64)) * inv_n).cos() +
                c.im * ((2.0 * PI * (k as f64) * (i as f64)) * inv_n).sin();
        }
        x.push(sum)
    }
    x
}

pub fn real_dft(x: &[f64]) -> Vec<c64> {
    if x.len() & 1 == 1 {
        panic!("x passed to real_dft must be have even len");
    }
    let mut fx = Vec::new();
    let n = x.len();
    let inv_n = 1.0 / n as f64;
    let scale_factor = inv_n * 2.0;
    // Zero REX[] and IMX[] so they can be used as accumulators
    for _ in 0..((n / 2) + 1) {
        fx.push(c64::new(0.0, 0.0));
    }
    for (k, c) in fx.iter_mut().enumerate() {
        for i in 0..n {
            c.re += x[i] * ((2.0 * PI * (k as f64) * (i as f64)) * inv_n).cos() * scale_factor;
            c.im += x[i] * ((2.0 * PI * (k as f64) * (i as f64)) * inv_n).sin() * scale_factor;
        }
    }
    fx
}   

#[cfg(test)]
mod tests {
    extern crate flot;
    extern crate rand;
    use num_traits::sign::Signed;
    use num_complex::Complex;
    use std::f64;
    use std::f32;
    use self::rand::Rng;
    use super::*;
    
    fn graph_complex(filename: &str, values: &[(&str, &[c64])]) {
        let page = flot::Page::new("");
        let p = page.plot("Graph").size(800, 400);
        for &(name, v) in values {
            let mut re_values = v.iter().enumerate().map(|(i, c)| (i as f64, c.re)).collect::<Vec<_>>();
            let mut im_values = v.iter().enumerate().map(|(i, c)| (i as f64, c.im)).collect::<Vec<_>>();
            p.lines(&(name.to_string() + ".re"), re_values);
            p.lines(&(name.to_string() + ".im"), im_values);
        }
        page.render(filename).unwrap();
    }

    fn reference_dft(input: &[c64]) -> Vec<c64> {
        let mut fft_input = Vec::from(&input[..]);
        let mut fft_output = vec![c64::zero(); fft_input.len()];
        let mut fft_planner = rustfft::FFTplanner::new(false);
        let fft = fft_planner.plan_fft(fft_input.len());
        fft.process(&mut fft_input, &mut fft_output);
        let scale_factor = 1.0 / (input.len() as f64).sqrt();
        for x in fft_output.iter_mut() {
            *x *= scale_factor;
        }
        fft_output
    }

    fn reference_idft(input: &[c64]) -> Vec<c64> {
        let mut fft_input = Vec::from(&input[..]);
        // for x in fft_input.iter_mut() {
        //     *x *= input.len() as f64;
        // }
        let mut fft_output = vec![c64::zero(); fft_input.len()];
        let mut fft_planner = rustfft::FFTplanner::new(true);
        let fft = fft_planner.plan_fft(fft_input.len());
        fft.process(&mut fft_input, &mut fft_output);
        let scale_factor = 1.0 / (input.len() as f64).sqrt();
        for x in fft_output.iter_mut() {
            *x *= scale_factor;
        }
        fft_output
    }

    fn complex_near_eq<T: Signed + std::cmp::PartialOrd + std::ops::Sub + Copy>(a: &[Complex<T>], b: &[Complex<T>], epsilon: T) -> bool {
        a.iter().zip(b).all(|(&Complex {re: ref xre, im: ref xim}, &Complex {re: ref yre, im: ref yim})|
        {
            (*xre - *yre).abs() < epsilon && (*xim - *yim).abs() < epsilon
        })
    }

    fn get_test_signal(len: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        (0..len)
            .map(|x| (x as f64 * PI * 2.0) / len as f64)
            .map(|x| 0.0
                     + x.cos()
                     + (x * 4.0).sin() * 0.5
                     + (x * 13.0).sin() * 2.0
                    //  + (if x > (2.0 * PI / 3.0) { 0.25 } else { -0.125 })
                    //  + rng.gen::<f64>() * 0.3
                )
            .collect::<Vec<_>>()
    }

    fn get_complex_test_signal(len: usize) -> Vec<c64> {
        get_test_signal(len)
            .into_iter()
            .map(|re| c64::new(re, 0.0))
            .collect::<Vec<_>>()
    }

    fn print_complex_sequence(prefix: &str, list: &[c64]) {
        print!("{}", prefix);
        for c in list.iter() {
            print!("{:.4} ", c);
        }
        println!("");
    }

    #[test]
    fn convolve_compare_time_domain_and_frequency_domain_implementation() {
        //let f = get_test_signal(14300);
        //let g = vec![1.0f64 / 136.0; 136]; // Just a moving average of length 136, to KISS.

        let f = vec![1.0, 2.0, 3.0, 4.0, -2.0, -6.0, -1.0, 44.0, 22.0];
        let g = vec![0.5, 0.25, 0.25, 0.5];

        // First do slow time domain convolution (well at the size it actually might be just as fast)
        let td_convolved = convolve_in_time_domain(&f, &g);
        println!();
        println!("td_convolved #0 {:?}", td_convolved);
        let fd_convolved = convolve_in_frequency_domain(&f, &g);
        println!("fd_convolved #0 {:?}", fd_convolved);
        assert_eq!(td_convolved.len(), f.len() + g.len() - 1);
        assert_eq!(fd_convolved.len(), f.len() + g.len() - 1);
        let epsilon = 1.0e-10;
        assert!(td_convolved.iter().zip(fd_convolved.iter()).any(|(a, b)| (*a - *b).abs() < epsilon));
    }

    #[test]
    fn graph_broken_up_signal() {
        let signal = [c64::new(1.0, 0.0), c64::new(-1.0, 0.0)]; // , c64::new(1.0, 0.0), c64::new(-1.0, 0.0)];
        let even: Vec<_> = signal.iter().enumerate().filter_map(|(i,x)| if i & 1 == 0 { Some(*x) } else { None }).collect();
        let odd: Vec<_> = signal.iter().enumerate().filter_map(|(i,x)| if i & 1 == 1 { Some(*x) } else { None }).collect();
        println!("signal:");
        print_complex_sequence("td", &signal);
        print_complex_sequence("fd", &complex_dft(&signal));
        println!("even:");
        print_complex_sequence("td", &even);
        print_complex_sequence("fd", &complex_dft(&even));
        println!("odd:");
        print_complex_sequence("td", &odd);
        print_complex_sequence("fd", &complex_dft(&odd));
        println!("combined:");
        let combined = fft_combine(&even, &odd);
        print_complex_sequence("", &combined);
    }

    #[test]
    fn naive_dft_both_ways() {
        let len = 64;
        let signal = (0..len).map(|x| (x as f64 * PI * 2.0) / len as f64)
            .map(|x| x.sin()).collect::<Vec<_>>();
        let sigfreq = real_dft(&signal);
        for (k, c) in sigfreq.iter().enumerate() {
            println!("{} {:.6}", k, c);
        }
        let converted = real_idft(&sigfreq);
        assert_eq!(len, converted.len());
        println!("naive_dft_both_ways");
        for i in signal.iter().zip(converted.iter()) {
            println!("{:.6} {:.6}", i.0, i.1);
        }
    }

    #[test]
    fn complex_dft_matches_reference_implementation() {
        let input = get_complex_test_signal(128);
        let epsilon = 1.0e-10;
        assert!(complex_near_eq(&reference_dft(&input), &complex_dft(&input), epsilon));
    }

    #[test]
    fn complex_fft_roundtrip() {
        let input = get_complex_test_signal(8);
        let epsilon = 1.0e-10;
        let output = complex_fft(&input);
        let reference_output = complex_dft(&input);
        println!("");
        print_complex_sequence("ref dft ", &reference_output);
        print_complex_sequence("    fft ", &output);

        let diff: Vec<_> = output.iter().zip(&reference_output).map(|(x, y)| *x - *y + c64::new(1.0, 1.0)).collect();
        graph_complex("complex_fft_roundtrip.html", &[("fft", &output),/*,("dft", &reference_output), */("diff", &diff)]);
        let rountripped = complex_ifft(&output);
        assert!(complex_near_eq(&input, &rountripped, epsilon));
    }

    #[test]
    fn complex_dft_roundtrip() {
        let input = get_complex_test_signal(128);
        let epsilon = 1.0e-10;
        let rountripped = complex_idft(&complex_dft(&input));
        assert!(complex_near_eq(&input, &rountripped, epsilon));
    }

    #[test]
    fn reference_dft_roundtrip() {
        let input = get_complex_test_signal(128);
        let epsilon = 1.0e-12;
        let rountripped = reference_idft(&reference_dft(&input));
        assert!(complex_near_eq(&input, &rountripped, epsilon));
    }

    #[test]
    fn naive_complex_dft_both_ways() {
        println!("complex_dft_both_ways");
        let len = 64;
        let signal = (0..len)
            .map(|x| (x as f64 * PI * 2.0) / len as f64)
            .map(|x| 0.0
                     + x.cos()
                     + (x * 2.3).sin() * 0.5
                     //+ (if x > (2.0 * PI / 3.0) { 0.25 } else { -0.125 })
                )
            .map(|x| c64::new(x, 0.0))
            .collect::<Vec<_>>();

        let fft_output = reference_dft(&signal);

        let sigfreq = complex_dft(&signal);
        let epsilon = 1.0e-10;
        assert!(complex_near_eq(&fft_output, &sigfreq, epsilon));
        for (k, (c, rustfft_c)) in sigfreq.iter().zip(fft_output).enumerate() {
            let (amp, rad) = c.to_polar();
            let deg = rad * 180.0 / PI;
            println!("{} {:.4}    {:.4}      {:.3}       {:.1}°", k, c, rustfft_c, amp, deg);
        }

        graph_complex("naive_complex_dft_both_ways", &[("x", &signal),("y", &sigfreq)]);
        let converted = complex_idft(&sigfreq);
        assert!(complex_near_eq(&converted, &signal, epsilon));
        assert_eq!(len, converted.len());
        for i in signal.iter().zip(converted.iter()) {
            println!("{:.6} {:.6}", i.0, i.1);
        }
    }
    #[test]
    fn some_complex_math() {
        let a = c64::new(1.0, 0.5);
        let b = c64::new(0.5, 1.0);
        println!("{} * {} = {}", a, b, a * b);
        println!("{}.norm() = {}", a, a.norm());
    }
}
