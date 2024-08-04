public class SampleClass2 {
    private String fieldA;

    public void methodA(@Nullable String paramA) {
        if (paramA == null) {
            System.out.println("paramA is null");
        }
    }
}

