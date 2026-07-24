
#[test]
fn ui_export() {
    let dir = std::path::Path::new("tests/ui-export");
    let has_cases = dir
        .read_dir()
        .map(|it| {
            it.filter_map(Result::ok)
                .any(|e| e.path().extension().map_or(false, |x| x == "rs"))
        })
        .unwrap_or(false);
    if has_cases {
        let t = trybuild::TestCases::new();
        t.compile_fail("tests/ui-export/*.rs");
    }
}
