macro_rules! raw_message {
    ($receiver:expr, $selector:literal $(, $argument:expr => $type:ty)*; $result:ty) => {{
        let receiver = $receiver;
        static SELECTOR: std::sync::OnceLock<objc2::runtime::Sel> = std::sync::OnceLock::new();
        let selector = *SELECTOR.get_or_init(|| objc2::runtime::Sel::register(obfstr::obfcstr!($selector)));
        let send: unsafe extern "C-unwind" fn(
            *const objc2::runtime::AnyObject,
            objc2::runtime::Sel,
            $($type),*
        ) -> $result = std::mem::transmute(objc2::ffi::objc_msgSend as *const ());
        send(
            std::ptr::from_ref(receiver).cast::<objc2::runtime::AnyObject>(),
            selector,
            $($argument),*
        )
    }};
}

macro_rules! raw_class {
    ($name:ident) => {
        #[repr(transparent)]
        #[derive(PartialEq, Eq, Hash)]
        pub struct $name(objc2::runtime::NSObject);

        unsafe impl objc2::encode::RefEncode for $name {
            const ENCODING_REF: objc2::encode::Encoding = objc2::encode::Encoding::Object;
        }

        unsafe impl objc2::Message for $name {}

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Debug::fmt(&self.0, f)
            }
        }
    };
    ($name:ident, $class:literal) => {
        raw_class!($name);

        impl $name {
            fn class() -> &'static objc2::runtime::AnyClass {
                static CLASS: std::sync::OnceLock<&'static objc2::runtime::AnyClass> =
                    std::sync::OnceLock::new();
                CLASS.get_or_init(|| {
                    objc2::runtime::AnyClass::get(obfstr::obfcstr!($class))
                        .expect("Objective-C class is unavailable")
                })
            }
        }
    };
}
