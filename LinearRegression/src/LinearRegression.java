import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import java.io.File;
import java.io.InputStreamReader;
import java.io.BufferedReader;
import java.io.FileInputStream;

public class LinearRegression {

    private double theta0 = 0.0;
    private double theta1 = 0.0;

    private double theta0p = 10.0;
    private double theta1p = 10.0;

    private double alpha = 0.01;

    private Double[][] data = new Double[100][2];//创建一个2行100列的二维数组（随便找100个数据测试）


    public LinearRegression() throws IOException{
        String pathname = "data/singleData.txt";
        File filename = new File(pathname);//获取文件句柄

        InputStreamReader reader = new InputStreamReader(
                new FileInputStream(filename));//把信息读取到内存中，并进行解读

        BufferedReader br = new BufferedReader(reader);//解读完成后进行输出，这一步是为了转换成IO可以识别的数据

        String line ;

        int j =0;

        while((line = br.readLine()) != null && j<100){

            String[] split = line.split(",");//对原来的txt格式数据进行分割

            data[j][0] = Double.valueOf(split[0]);//分割后把数据存入数组中
            data[j][1] = Double.valueOf(split[1]);

            j++;
        }


    }

    public double predict(double x){
        return  theta0 + theta1*x;//最简单的一元线性回归的假设函数
    }

    public double calc_error(double x, double y){
        return predict(x)-y;//注意，这里的计算已经进行了简化了，这里并不是真正的损失函数
        //损失函数应该要平方，可是在后续的求导过程中，发现只要直接相减就足够了，所以这里为了简化计算直接计算差值
    }

    public void gradientDescient(){//这个梯度下降法要修改一下
        double sum0 = 0.0;
        double sum1 = 0.0;

        for(int i =0; i<97; i++){//求的是总和

            sum0 += calc_error(data[i][0], data[i][1]);//这里只是损失函数对x的系数求偏导并简化后的其中一项，真正的损失函数应该是均方误差
            sum1 += calc_error(data[i][0], data[i][1])*data[i][0];//这里是损失函数对截距求偏导并简化后的其中一项
        }

        this.theta0 = theta0 - alpha*sum0/data.length;//对截距和系数进行迭代
        this.theta1 = theta1 - alpha*sum1/data.length;

    }

    //开始迭代，利用梯度下降法不断更新参数，直到两次迭代的参数之间的差小于误差值
    public void lineGre(){
        int itea = 0;
        while(Math.abs(theta1p-theta1)>0.0001 || Math.abs(theta0p-theta0)>0.0001){

            System.out.println("the current step is :" + itea);
            System.out.println("theta0 " + theta0);
            System.out.println("theta1 " + theta1);
            System.out.println("theta1p "+ theta1p);
            System.out.println("theta1 "+ theta1);
            System.out.println("theta0p "+ theta0p);
            System.out.println("theta0 "+ theta0);

            System.out.println();

            theta0p = theta0;
            theta1p = theta1;

            gradientDescient();
            itea++;
        }
    }

    public static void main(String[] args) throws IOException{

        LinearRegression linearRegression = new LinearRegression();

        linearRegression.lineGre();

        List<Double> list = new ArrayList<Double>();

        for(int i = 0; i< linearRegression.data.length; i++){
            list.add(linearRegression.data[i][0]);
        }

    }

}
