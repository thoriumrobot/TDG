public class SampleClass1 {
    private String field1;
    private Integer field2;

    public SampleClass1(String field1, Integer field2) {
        this.field1 = field1;
        this.field2 = field2;
    }

    public void method1(@Nullable String param1) {
        if (param1 == null) {
            System.out.println("param1 is null");
        }
    }

    public Integer method2(Integer param2) {
        return param2 != null ? param2 : 0;
    }
}

