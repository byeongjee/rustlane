#[test]
fn ui_derive() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui-derive/*.rs");
}
