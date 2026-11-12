//! JSON as Python's `json` module reads and writes it: an object keeps its keys in file order, an
//! integer stays an integer and a float a float, and `dumps` writes Python's `json.dumps` text (its
//! defaults: `", "` / `": "` separators, `ensure_ascii`, floats as `repr(float)`). A run's settings
//! reach the game's spec through `zero.games.spec_json` (a dict `json.dumps`'d), so a native worker
//! reproduces that text byte for byte (docs/az.md 42).
use std::fmt::Write;

use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub enum PyJson {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    List(Vec<PyJson>),
    Dict(Vec<(String, PyJson)>),
}

impl<'de> Deserialize<'de> for PyJson {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<PyJson, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = PyJson;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("JSON")
            }
            fn visit_unit<E>(self) -> Result<PyJson, E> {
                Ok(PyJson::Null)
            }
            fn visit_none<E>(self) -> Result<PyJson, E> {
                Ok(PyJson::Null)
            }
            fn visit_bool<E>(self, b: bool) -> Result<PyJson, E> {
                Ok(PyJson::Bool(b))
            }
            fn visit_i64<E>(self, i: i64) -> Result<PyJson, E> {
                Ok(PyJson::Int(i as i128))
            }
            fn visit_u64<E>(self, u: u64) -> Result<PyJson, E> {
                Ok(PyJson::Int(u as i128))
            }
            fn visit_f64<E>(self, f: f64) -> Result<PyJson, E> {
                Ok(PyJson::Float(f))
            }
            fn visit_str<E>(self, s: &str) -> Result<PyJson, E> {
                Ok(PyJson::Str(s.to_string()))
            }
            fn visit_string<E>(self, s: String) -> Result<PyJson, E> {
                Ok(PyJson::Str(s))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut a: A) -> Result<PyJson, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = a.next_element()? {
                    v.push(x);
                }
                Ok(PyJson::List(v))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut a: A) -> Result<PyJson, A::Error> {
                let mut v: Vec<(String, PyJson)> = Vec::new();
                while let Some((k, x)) = a.next_entry::<String, PyJson>()? {
                    // (a repeated key: Python's dict keeps the last value at the first key's place)
                    if let Some(e) = v.iter_mut().find(|e| e.0 == k) {
                        e.1 = x;
                    } else {
                        v.push((k, x));
                    }
                }
                Ok(PyJson::Dict(v))
            }
        }
        d.deserialize_any(V)
    }
}

impl PyJson {
    pub fn parse(text: &str) -> Result<PyJson, String> {
        serde_json::from_str(text).map_err(|e| format!("JSON: {e}"))
    }

    pub fn get(&self, k: &str) -> Option<&PyJson> {
        match self {
            PyJson::Dict(v) => v.iter().find(|e| e.0 == k).map(|e| &e.1),
            _ => None,
        }
    }

    /// As a serde_json value (keys sorted; integers beyond i64 / u64 are refused).
    pub fn to_value(&self) -> Result<Value, String> {
        Ok(match self {
            PyJson::Null => Value::Null,
            PyJson::Bool(b) => Value::Bool(*b),
            PyJson::Int(i) => {
                if let Ok(x) = i64::try_from(*i) {
                    Value::from(x)
                } else if let Ok(x) = u64::try_from(*i) {
                    Value::from(x)
                } else {
                    return Err(format!("integer {i} out of range"));
                }
            }
            PyJson::Float(f) => serde_json::Number::from_f64(*f).map(Value::Number).ok_or_else(|| format!("{f}: not a finite number"))?,
            PyJson::Str(s) => Value::String(s.clone()),
            PyJson::List(v) => Value::Array(v.iter().map(|x| x.to_value()).collect::<Result<_, _>>()?),
            PyJson::Dict(v) => Value::Object(v.iter().map(|(k, x)| Ok((k.clone(), x.to_value()?))).collect::<Result<_, String>>()?),
        })
    }

    /// Python's `json.dumps(self)` (its default arguments).
    pub fn dumps(&self) -> String {
        let mut s = String::new();
        self.dump_into(&mut s);
        s
    }

    fn dump_into(&self, s: &mut String) {
        match self {
            PyJson::Null => s.push_str("null"),
            PyJson::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            PyJson::Int(i) => {
                let _ = write!(s, "{i}");
            }
            PyJson::Float(f) => s.push_str(&py_float_repr(*f)),
            PyJson::Str(t) => dump_str(t, s),
            PyJson::List(v) => {
                s.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    x.dump_into(s);
                }
                s.push(']');
            }
            PyJson::Dict(v) => {
                s.push('{');
                for (i, (k, x)) in v.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    dump_str(k, s);
                    s.push_str(": ");
                    x.dump_into(s);
                }
                s.push('}');
            }
        }
    }
}

/// A string as `json.dumps` writes it with `ensure_ascii` (`\uXXXX` for every non-ASCII character,
/// surrogate pairs beyond the BMP).
fn dump_str(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            '\u{8}' => s.push_str("\\b"),
            '\u{c}' => s.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) {
                    let _ = write!(s, "\\u{u:04x}");
                }
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

/// `repr(f)` for a finite float (Python's shortest round-trip digits; exponent form when the decimal
/// exponent is < -4 or >= 16), `NaN` / `Infinity` / `-Infinity` as `json.dumps` writes them.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "NaN".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` is the shortest round-trip representation: d[.ddd]e<exp>
    let e = format!("{:e}", f.abs());
    let (mant, exp) = e.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f < 0.0 { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        return format!("{sign}{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs());
    }
    let n = digits.len() as i32;
    if exp >= 0 {
        let int_len = exp + 1;
        if n <= int_len {
            format!("{sign}{}{}.0", digits, "0".repeat((int_len - n) as usize))
        } else {
            format!("{sign}{}.{}", &digits[..int_len as usize], &digits[int_len as usize..])
        }
    } else {
        format!("{sign}0.{}{}", "0".repeat((-exp - 1) as usize), digits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_is_pythons() {
        for (f, want) in [(1e16, "1e+16"), (1e15, "1000000000000000.0"), (0.0001, "0.0001"), (0.00001, "1e-05"), (123456789012345678.0, "1.2345678901234568e+17"), (0.5, "0.5"), (2.0, "2.0"), (-1.5e-7, "-1.5e-07"), (1.0 / 3.0, "0.3333333333333333"), (20.0, "20.0"), (1e22, "1e+22"), (123.456, "123.456"), (5e-324, "5e-324")] {
            assert_eq!(py_float_repr(f), want, "{f}");
        }
    }

    #[test]
    fn dumps_is_pythons() {
        let j = PyJson::parse(r#"{"sealed": {"set": "eoe", "best_share": 0.5, "n": [1, 2.0, true, null]}, "max_turns": 60, "é": "ü\n𝄞"}"#).unwrap();
        // (Python: json.dumps(json.loads(that)) - ensure_ascii escapes, a surrogate pair beyond the BMP)
        let want = "{\"sealed\": {\"set\": \"eoe\", \"best_share\": 0.5, \"n\": [1, 2.0, true, null]}, \"max_turns\": 60, \"\\u00e9\": \"\\u00fc\\n\\ud834\\udd1e\"}";
        assert_eq!(j.dumps(), want);
    }
}
