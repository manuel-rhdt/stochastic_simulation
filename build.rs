use vergen::{ConstantsFlags, generate_cargo_keys};

fn main() {
    generate_cargo_keys(ConstantsFlags::all()).expect("Unable to generate the cargo keys!");
}